"""Differentiable DRR renderer reproducing Autoscoper's ray caster (RayCaster.cl).

Autoscoper, per pixel:  DRR = clamp( sum_samples (val > cutoff) * step * val / intensity, 0, 1 )
with val read from the UNORM16 volume (raw / 65535) by trilinear interpolation, `step` the sample
distance in mm, and intensity = exp(15 * RayIntensity - 5), step = exp(7 * SampleDistance - 5)
(ModelViewTreeWidgetItem.cpp). Only the region covered by the projected bounding box of the bone's
non-zero voxels is rendered (Tracker::calculate_viewport), at render pixel spacing `px` image pixels.

To keep memory and time down on the CPU, each volume is cropped to its non-zero bounding box (+2
voxels) and each ray is sampled only between its entry and exit of that box.
"""
import numpy as np
import tifffile
import torch
import torch.nn.functional as F


class Bone:
    def __init__(self, path, voxel, name=None, cutoff_frac=0.0, margin=2):
        v = tifffile.imread(path)
        self.name = name
        self.D0, self.H0, self.W0 = v.shape              # full volume dims define the local frame
        self.sx, self.sy, self.sz = voxel
        k, r, c = np.nonzero(v)
        self.k0, self.r0, self.c0 = [max(0, x.min() - margin) for x in (k, r, c)]
        k1, r1, c1 = k.max() + 1 + margin, r.max() + 1 + margin, c.max() + 1 + margin
        crop = v[self.k0:k1, self.r0:r1, self.c0:c1].astype(np.float32) / 65535.0
        self.D, self.H, self.W = crop.shape
        self.vol = torch.from_numpy(np.ascontiguousarray(crop))[None, None]
        self.vmax = float(crop.max())
        self.cutoff = cutoff_frac * self.vmax             # Autoscoper: fraction of [min, max] (min = 0)
        # non-zero bounding box in local mm (x = col*sx, y = (H0-row)*sy, z = -slice*sz)
        xs = np.array([c.min(), c.max() + 1]) * self.sx
        ys = (self.H0 - np.array([r.max() + 1, r.min()])) * self.sy
        zs = -np.array([k.max() + 1, k.min()]) * self.sz
        self.box_lo = np.array([xs[0], ys[0], zs[0]]); self.box_hi = np.array([xs[1], ys[1], zs[1]])
        self.corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
        self.pivot = self.corners.mean(0)                 # Autoscoper default pivot: bbox centre
        P = np.c_[(c + .5) * self.sx, (self.H0 - r - .5) * self.sy, -(k + .5) * self.sz]
        self.centroid = P.mean(0)
        # principal axes of the voxel cloud (rows: long, mid, short)
        w, V = np.linalg.eigh(np.cov((P - self.centroid).T))
        self.axes = V[:, ::-1].T
        self._mask_vox = None

    def sample(self, P_local):
        """P_local (..., 3) mm in the volume frame -> voxel values (trilinear, zero outside the crop)."""
        col = P_local[..., 0] / self.sx - self.c0
        row = self.H0 - P_local[..., 1] / self.sy - self.r0
        sl = -P_local[..., 2] / self.sz - self.k0
        g = torch.stack([2 * col / self.W - 1, 2 * row / self.H - 1, 2 * sl / self.D - 1], -1)
        out = F.grid_sample(self.vol, g.reshape(1, 1, 1, -1, 3).to(self.vol.dtype), mode='bilinear',
                            padding_mode='zeros', align_corners=False)
        return out.reshape(P_local.shape[:-1])

    def surface_points(self, n=2500, quantile=0.6, seed=0):
        """Points on strong 3D intensity gradients (cortical outline), local mm."""
        v = self.vol[0, 0].numpy()
        gz, gy, gx = np.gradient(v, self.sz, self.sy, self.sx); g = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        inside = v > max(self.cutoff, 1e-9)
        k, r, c = np.nonzero(inside & (g > np.quantile(g[inside], quantile)))
        sel = np.random.default_rng(seed).choice(len(k), size=min(n, len(k)), replace=False)
        return np.c_[(c[sel] + self.c0 + .5) * self.sx, (self.H0 - r[sel] - self.r0 - .5) * self.sy,
                     -(k[sel] + self.k0 + .5) * self.sz]


class View:
    """One camera, rendering at `render_res` pixels across the full image width."""

    def __init__(self, cam, render_res=512, step=0.5, intensity=10.0):
        self.cam, self.step, self.intensity = cam, step, intensity
        self.render_res = render_res
        self.px = cam.W / render_res                       # image pixels per render pixel
        self.Rc = torch.tensor(cam.R); self.C = torch.tensor(cam.centre)

    def roi(self, bone, R, t):
        """Projected bounding box of the bone's non-zero voxels, snapped to the render grid."""
        P = bone.corners @ np.asarray(R.detach() if torch.is_tensor(R) else R).T + \
            np.asarray(t.detach() if torch.is_tensor(t) else t)
        uv = self.cam.project(P)
        lo = np.floor(np.clip(uv.min(0), 0, [self.cam.W, self.cam.H]) / self.px) * self.px
        hi = np.ceil(np.clip(uv.max(0), 0, [self.cam.W, self.cam.H]) / self.px) * self.px
        return lo, hi

    def union_roi(self, rois):
        lo = np.min([r[0] for r in rois], 0); hi = np.max([r[1] for r in rois], 0)
        return lo, hi

    def pixel_grid(self, lo, hi):
        cols = torch.arange(lo[0], hi[0] - 1e-6, self.px, dtype=torch.float64) + self.px / 2
        rows = torch.arange(lo[1], hi[1] - 1e-6, self.px, dtype=torch.float64) + self.px / 2
        return torch.meshgrid(rows, cols, indexing='ij')

    def _rays(self, lo, hi):
        rr, cc = self.pixel_grid(lo, hi)
        d = torch.from_numpy(self.cam.ray_dirs_cam(cc.numpy(), rr.numpy())) @ self.Rc   # world dirs
        return d / d.norm(dim=-1, keepdim=True)

    def render(self, bones_poses, lo, hi, rays=None):
        """DRR over the ROI for a list of (Bone, R, t); summed before clamping. Differentiable in R, t."""
        return torch.clamp(sum(self.render_parts(bones_poses, lo, hi, rays)), 0, 1)

    def render_parts(self, bones_poses, lo, hi, rays=None):
        """Unclamped DRR of each (Bone, R, t) over the ROI (list of (h, w) tensors).

        Samples along each ray are spaced `step` mm, starting half a step inside the ray's entry into
        the bone's non-zero bounding box (Autoscoper marches the whole unit box from the far side; the
        Riemann sums agree to within the sample spacing). Sample positions are built in float32
        directly in normalised grid coordinates to save memory and time."""
        d = self._rays(lo, hi) if rays is None else rays
        hw = d.shape[:2]
        df = d.reshape(-1, 3)
        parts = []
        for bone, R, t in bones_poses:
            img = torch.zeros(df.shape[0], dtype=torch.float64)
            parts.append(img)
            with torch.no_grad():
                Rd = R.detach()
                o_l = (self.C - t.detach()) @ Rd                # camera centre in the local frame
                d_l = df @ Rd
                inv = 1.0 / torch.where(d_l.abs() < 1e-12, torch.full_like(d_l, 1e-12), d_l)
                t0 = (torch.tensor(bone.box_lo) - o_l) * inv; t1 = (torch.tensor(bone.box_hi) - o_l) * inv
                tn = torch.minimum(t0, t1).amax(-1); tf = torch.maximum(t0, t1).amin(-1)
                idx = torch.nonzero(tf > tn)[:, 0]              # only rays that cross the bone's box
                if len(idx) == 0:
                    continue
                span = tf[idx] - tn[idx]
                order = torch.argsort(span)
                buckets = torch.tensor_split(order, min(4, len(order)))   # rays grouped by path length
            # local point = o + s * dl with o = R^T (C - t), dl = R^T d  (differentiable in R, t)
            o = (self.C - t) @ R
            # affine map local mm -> normalised grid coords of the cropped volume
            A = torch.tensor([2 / (bone.sx * bone.W), -2 / (bone.sy * bone.H), -2 / (bone.sz * bone.D)],
                             dtype=torch.float64)
            B = torch.tensor([-2 * bone.c0 / bone.W - 1, 2 * (bone.H0 - bone.r0) / bone.H - 1,
                              -2 * bone.k0 / bone.D - 1], dtype=torch.float64)
            og = (o * A + B).float()
            for bk in buckets:
                ri = idx[bk]
                with torch.no_grad():
                    n = int(np.ceil(float(span[bk].max()) / self.step))
                    s = tn[ri, None] + (torch.arange(n, dtype=torch.float64) + 0.5) * self.step   # r,n
                    valid = s < tf[ri, None]
                    s = s.float()
                dg = ((df[ri] @ R) * A).float()
                g = og + s[..., None] * dg[:, None, :]                                          # r,n,3
                v = F.grid_sample(bone.vol, g.reshape(1, 1, 1, -1, 3), mode='bilinear', padding_mode='zeros',
                                  align_corners=False).reshape(s.shape)
                v = torch.where(valid & (v > bone.cutoff), v, torch.zeros_like(v))
                img = img.index_add(0, ri, v.sum(-1).double() * (self.step / self.intensity))
            parts[-1] = img
        return [p.reshape(hw) for p in parts]

    def sample_image(self, im, lo, hi, mode='bilinear'):
        """Sample an (h, w) image (any resolution covering the full field) at the ROI's render pixels."""
        rr, cc = self.pixel_grid(lo, hi)
        g = torch.stack([2 * cc / self.cam.W - 1, 2 * rr / self.cam.H - 1], -1).to(im.dtype)
        return F.grid_sample(im[None, None], g[None], mode=mode, align_corners=False)[0, 0]
