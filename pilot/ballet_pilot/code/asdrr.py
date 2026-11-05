"""Differentiable re-implementation of Autoscoper's per-frame cost on the CPU (PyTorch).

Mirrors libautoscoper:
  * camera model      : MayaCam 1.0 (ascam.py, port of Camera.cpp / CoordFrame.cpp)
  * volume local frame: x = col*sx, y = (H-row)*sy, z = -slice*sz  (VolumeDescription / RayCaster.cl)
  * DRR               : sum over ray samples of (val>cutoff)*val*step / intensity, clamped to [0,1]
  * ROI               : projected bounding box of the non-zero voxels, sampled at RenderResolution
  * cost              : sum over cameras of (1 - NCC) over pixels where DRR != 0
Filters are not applied (identity filter chain).
"""
import numpy as np, torch, torch.nn.functional as F, tifffile

torch.set_num_threads(2)


def skew(v):
    z = torch.zeros((), dtype=v.dtype)
    return torch.stack([torch.stack([z, -v[2], v[1]]),
                        torch.stack([v[2], z, -v[0]]),
                        torch.stack([-v[1], v[0], z])])


def rotvec_to_matrix(rv_deg):
    """Rodrigues; rotation vector in degrees (Autoscoper's increment convention). Differentiable."""
    rv = rv_deg * (np.pi / 180.0)
    th = torch.sqrt((rv ** 2).sum() + 1e-18)
    K = skew(rv / th)
    return torch.eye(3, dtype=rv.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


class Bone:
    def __init__(self, path, voxel=(0.625, 0.625, 1.25), cutoff_frac=0.03):
        v = tifffile.imread(path).astype(np.float32) / 65535.0  # (D,H,W), UNORM16 like OpenCL
        self.D, self.H, self.W = v.shape
        self.sx, self.sy, self.sz = voxel
        self.vol = torch.tensor(v)[None, None]  # N,C,D,H,W
        self.cutoff = float(v.min() + cutoff_frac * (v.max() - v.min()))  # UI slider mapping
        k, r, c = np.nonzero(v)
        # non-zero bounding box corners in local mm
        xs = np.array([c.min(), c.max() + 1]) * self.sx
        ys = (self.H - np.array([r.max() + 1, r.min()])) * self.sy
        zs = -np.array([k.max() + 1, k.min()]) * self.sz
        self.corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
        self.centre = self.corners.mean(0)          # Autoscoper's default pivot (bbox centre)
        self.radius = np.linalg.norm(self.corners - self.centre, axis=1).max()
        # voxel-weighted centroid & principal axes (handy for interpretation)
        P = np.c_[(c + .5) * self.sx, (self.H - r - .5) * self.sy, -(k + .5) * self.sz]
        self.centroid = P.mean(0)

    def sample(self, P_local):
        """P_local: (...,3) mm in volume frame -> voxel values (trilinear, zero outside)."""
        g = torch.stack([2 * P_local[..., 0] / (self.sx * self.W) - 1,
                         2 * (self.H - P_local[..., 1] / self.sy) / self.H - 1,
                         2 * (-P_local[..., 2] / self.sz) / self.D - 1], -1)
        out = F.grid_sample(self.vol, g.reshape(1, 1, 1, -1, 3), mode='bilinear',
                            padding_mode='zeros', align_corners=False)
        return out.reshape(P_local.shape[:-1])


class Pose:
    """Bone pose: world = R @ p_local + t. Increments are applied about the pivot, in the bone frame,
    exactly like Autoscoper's  new = current * V^-1 * Delta * V."""
    def __init__(self, R, t, pivot):
        self.R = torch.as_tensor(R, dtype=torch.float64)
        self.t = torch.as_tensor(t, dtype=torch.float64)
        self.pivot = torch.as_tensor(pivot, dtype=torch.float64)

    def apply(self, delta):
        """delta: tensor(6) = [dx,dy,dz (mm), rotation vector (deg)] -> (R,t) differentiable."""
        dR = rotvec_to_matrix(delta[3:])
        R = self.R @ dR
        # rotate about pivot: p -> pivot + dR (p - pivot) + dt  (in bone frame)
        t = self.t + self.R @ (self.pivot - dR @ self.pivot + delta[:3])
        return R, t

    def compose(self, delta):
        with torch.no_grad():
            R, t = self.apply(torch.as_tensor(delta, dtype=torch.float64))
        return Pose(R, t, self.pivot)


class View:
    def __init__(self, cam, render_res=512, step=0.5, intensity=10.0):
        self.cam, self.step, self.intensity = cam, step, intensity
        self.px = cam.W / render_res                 # image pixels per render pixel
        self.Rc = torch.tensor(cam.R); self.tc = torch.tensor(cam.t)

    def roi(self, bone, R, t):
        P = bone.corners @ R.detach().numpy().T + t.detach().numpy()
        uv = self.cam.project(P)
        lo = np.clip(uv.min(0), 0, [self.cam.W, self.cam.H]); hi = np.clip(uv.max(0), 0, [self.cam.W, self.cam.H])
        return lo, hi

    def pixel_grid(self, lo, hi):
        cols = torch.arange(lo[0], hi[0], self.px, dtype=torch.float64) + self.px / 2
        rows = torch.arange(lo[1], hi[1], self.px, dtype=torch.float64) + self.px / 2
        return torch.meshgrid(rows, cols, indexing='ij')

    def render(self, bones_poses, lo, hi, n_extra=0):
        """DRR over ROI for one or several (bone, R, t). Differentiable wrt R,t."""
        rr, cc = self.pixel_grid(lo, hi)
        c = self.cam
        d_cam = torch.stack([(cc - c.cx) / c.f, (c.H - c.cy - rr) / c.f, -torch.ones_like(cc)], -1)
        d = d_cam @ self.Rc                          # world directions (rows of R are axes)
        d = d / d.norm(dim=-1, keepdim=True)
        img = 0
        for bone, R, t in bones_poses:
            ctr = R @ torch.tensor(bone.centre) + t
            dist = ((ctr - self.tc) * d).sum(-1)     # along-ray distance to bone centre
            n = int(np.ceil(2 * bone.radius / self.step))
            s = (torch.arange(n, dtype=torch.float64) - n / 2) * self.step
            P = self.tc + (dist[..., None] + s)[..., None] * d[..., None, :]   # h,w,n,3 world
            Pl = (P - t) @ R                          # world -> local (R orthonormal)
            v = bone.sample(Pl.float())
            v = torch.where(v > bone.cutoff, v, torch.zeros_like(v))
            img = img + v.sum(-1) * self.step / self.intensity
        return torch.clamp(img, 0, 1)

    def radiograph(self, rad, lo, hi):
        """Sample an (H,W) radiograph [0,1] over the ROI and invert, like RadRenderer.cl."""
        rr, cc = self.pixel_grid(lo, hi)
        g = torch.stack([2 * cc / self.cam.W - 1, 2 * rr / self.cam.H - 1], -1).float()
        out = F.grid_sample(rad[None, None], g[None], mode='bilinear', align_corners=False)[0, 0]
        return 1 - out


def masked_ncc(drr, rad, mask):
    n = mask.sum()
    f = drr * mask; g = rad * mask
    mf, mg = f.sum() / n, g.sum() / n
    a = (f - mf) * mask; b = (g - mg) * mask
    return (a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum() + 1e-12)


def cost(views, rads, bone, pose, delta, others=(), joint=False):
    """Autoscoper cost: sum_cam (1 - NCC). `others`: list of (bone, Pose) also present.
    joint=False -> only `bone` is rendered (Autoscoper behaviour); joint=True -> all rendered."""
    R, t = pose.apply(delta)
    total = 0
    for v, rad in zip(views, rads):
        lo, hi = v.roi(bone, R, t)
        bp = [(bone, R, t)]
        if joint:
            bp += [(b, p.R, p.t) for b, p in others]
        drr = v.render(bp, lo, hi)
        own = v.render([(bone, R, t)], lo, hi) if joint else drr
        mask = (own.detach() != 0).float()
        total = total + 1 - masked_ncc(drr, v.radiograph(rad, lo, hi), mask)
    return total
