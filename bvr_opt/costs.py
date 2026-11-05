"""Image similarity terms.

autoscoper_cost   Replica of Autoscoper's per-frame cost: sum over cameras of 1 - NCC between the
                  filtered DRR of one bone and the filtered, inverted radiograph, over the pixels
                  where the filtered DRR is non-zero (Tracker::trackFrame, Ncc.cl).
gradcorr_cost     Proposed DRR term: signed gradient correlation (mean NCC of x- and y-Sobel images)
                  after a Gaussian blur (sigma = 1 render pixel), inside the bone's silhouette dilated by
                  3 render pixels and the eroded field-of-view mask. Co-fitted bones are rendered together.
F2F               Model-based frame-to-frame term (see class docstring).
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter, sobel as nd_sobel

from .filters import apply_chain, blur, sobel_xy


def masked_ncc(f, g, mask):
    n = mask.sum().clamp_min(1)
    mf, mg = (f * mask).sum() / n, (g * mask).sum() / n
    a = (f - mf) * mask; b = (g - mg) * mask
    return (a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum() + 1e-12)


def fov_masks(trial, thresh=0.02, erode=15):
    """Field-of-view (image intensifier) mask per camera: per-pixel maximum over the trial above
    `thresh`, holes filled, eroded by `erode` pixels. Returns list of float32 tensors (H, W)."""
    out = []
    for ci in range(len(trial.cams)):
        mx = np.zeros((trial.cams[ci].H, trial.cams[ci].W), np.float32)
        for f in range(trial.n_frames):
            np.maximum(mx, trial.raw_image(ci, f), out=mx)
        m = binary_erosion(binary_fill_holes(mx > thresh), iterations=erode)
        out.append(torch.from_numpy(m.astype(np.float32)))
    return out


def sample_mask(view, mask, lo, hi):
    return view.sample_image(mask, lo, hi, mode='nearest') > 0.5


# ------------------------------------------------------------------------- Autoscoper replica
def autoscoper_cost(views, rads, bone, pose, delta, vie, per_cam=False):
    """rads: per-camera radiographs (raw, [0,1], any resolution). Autoscoper inverts them (RadRenderer)."""
    R, t = pose.apply(delta)
    cs = []
    for v, rad in zip(views, rads):
        lo, hi = v.roi(bone, R, t)
        if hi[0] - lo[0] < v.px or hi[1] - lo[1] < v.px:
            cs.append(torch.tensor(9999.0, dtype=torch.float64)); continue   # not in view
        drr = apply_chain(v.render([(bone, R, t)], lo, hi), vie['drr'])
        r = apply_chain(1 - v.sample_image(rad, lo, hi).double(), vie['rad'])
        mask = (drr.detach() != 0).double()
        cs.append(1 - masked_ncc(drr, r, mask))
    return cs if per_cam else sum(cs)


# ------------------------------------------------------------------------- proposed DRR term
def gradcorr(d, r, m):
    dx, dy = sobel_xy(d); rx, ry = sobel_xy(r)
    return 0.5 * (masked_ncc(dx, rx, m) + masked_ncc(dy, ry, m))


def gradcorr_cost(views, rads, fovs, bone, R, t, others=(), dil=3):
    """bone at (R, t) (differentiable); others: [(Bone, R, t)] rendered into the same DRR (co-fitting).
    Returns the sum over cameras of 1 - gradient correlation."""
    total = 0
    for v, rad, fov in zip(views, rads, fovs):
        lo, hi = v.roi(bone, R, t)
        if hi[0] - lo[0] < 3 * v.px or hi[1] - lo[1] < 3 * v.px:
            continue
        rays = v._rays(lo, hi)
        own = v.render([(bone, R, t)], lo, hi, rays)
        drr = v.render([(bone, R, t)] + list(others), lo, hi, rays) if others else own
        m = (own.detach() > 0).float()[None, None]
        mask = F.max_pool2d(m, 2 * dil + 1, 1, dil)[0, 0].double() * sample_mask(v, fov, lo, hi).double()
        if mask.sum() < 20:
            continue
        r = blur(1 - v.sample_image(rad, lo, hi).double(), 1.0)
        total = total + 1 - gradcorr(blur(drr, 1.0), r, mask)
    return total


# ------------------------------------------------------------------------- frame-to-frame term
class F2F:
    """Model-based frame-to-frame (optical-flow-like) term.

    Points on strong CT gradients of each bone are placed with pose(f) and projected into image f, and
    with pose(f+1) into image f+1, in both cameras. Signed gradient images of the blurred, inverted
    radiographs (sigma = 2 full-resolution pixels) are sampled at those points and compared by NCC.
    A correct pair of poses makes the same anatomy land on the same image structure in both frames.
    Cost = 1 - mean NCC over consecutive pairs and cameras."""

    def __init__(self, trial, frames, fovs, sigma=2.0, crop=None):
        """crop: per-camera (c0, r0, c1, r1) pixel box to keep (saves memory); default whole image."""
        self.trial, self.fovs = trial, fovs
        self.cams = [(torch.tensor(c.R), torch.tensor(c.t), c) for c in trial.cams]
        self.crop = crop or [(0, 0, c.W, c.H) for c in trial.cams]
        self.G = {}
        for f in frames:
            for ci in range(len(trial.cams)):
                c0, r0, c1, r1 = self.crop[ci]
                im = gaussian_filter(1 - trial.raw_image(ci, f), sigma)[r0:r1, c0:c1]
                self.G[(f, ci)] = torch.from_numpy(np.stack([nd_sobel(im, axis=1), -nd_sobel(im, axis=0)]))[None].float()

    def project(self, ci, P):
        Rc, tc, c = self.cams[ci]
        pc = P @ Rc.T + tc
        return c.fx * pc[:, 0] / pc[:, 2] + c.cx + 0.5, c.fy * pc[:, 1] / pc[:, 2] + c.cy + 0.5

    def sample(self, f, ci, col, row):
        c = self.cams[ci][2]; c0, r0, c1, r1 = self.crop[ci]
        gc = torch.stack([2 * (col - c0) / (c1 - c0) - 1, 2 * (row - r0) / (r1 - r0) - 1], -1)[None, None].float()
        val = F.grid_sample(self.G[(f, ci)], gc, mode='bilinear', align_corners=False)[0, :, 0].double()
        g = torch.stack([2 * col / c.W - 1, 2 * row / c.H - 1], -1)[None, None].float()
        inside = F.grid_sample(self.fovs[ci][None, None], g, mode='nearest', align_corners=False)[0, 0, 0] > 0.5
        inside = inside & (gc[0, 0].abs() < 1).all(-1)
        return val, inside

    @staticmethod
    def _ncc(x, y):
        x = x - x.mean(); y = y - y.mean()
        return (x * y).sum() / torch.sqrt((x * x).sum() * (y * y).sum() + 1e-12)

    def pair(self, P0, P1, f):
        """P0, P1: world points of the same bone points at frames f and f+1."""
        tot, n = 0, 0
        for ci in range(len(self.cams)):
            v0, i0 = self.sample(f, ci, *self.project(ci, P0))
            v1, i1 = self.sample(f + 1, ci, *self.project(ci, P1))
            m = i0 & i1
            if m.sum() < 50:
                continue
            tot = tot + 1 - 0.5 * (self._ncc(v0[0][m], v1[0][m]) + self._ncc(v0[1][m], v1[1][m])); n += 1
        return tot, n
