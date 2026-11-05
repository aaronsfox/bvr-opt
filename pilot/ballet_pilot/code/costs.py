import math, torch
import torch.nn.functional as F
from asdrr import View, masked_ncc
from filters import apply_chain, parse_vie

VIE = parse_vie('/mnt/user-data/attachments/de80d863-0bfc-5776-b7cf-80da25e9c8d8/4b52162c-control_settings.vie')
INTENSITY = math.exp(15 * VIE['renderer']['RayIntensity'] - 5)     # UI slider -> value (ModelViewTreeWidgetItem)


USE_FOV = False   # set True to exclude pixels outside the image intensifier circle (Autoscoper background-threshold equivalent)
import numpy as _np
_FOV = None
def _fov_mask(ci, v, lo, hi):
    global _FOV
    if _FOV is None:
        _FOV = [torch.tensor(_np.load(f'/root/work/fov_c{i+1}.npy').astype(_np.float32)) for i in range(2)]
    rr, cc = v.pixel_grid(lo, hi)
    g = torch.stack([2 * cc / v.cam.W - 1, 2 * rr / v.cam.H - 1], -1).float()
    return (F.grid_sample(_FOV[ci][None, None], g[None], mode='nearest', align_corners=False)[0, 0] > 0.5)


def make_views(cams, res=512):
    return [View(c, res, step=0.5, intensity=INTENSITY) for c in cams]


def autoscoper_pre(rad_chain=VIE['rad'], drr_chain=VIE['drr']):
    """Returns (rad_fn, drr_fn) applying Autoscoper filter chains."""
    return (lambda im: apply_chain(im, rad_chain)), (lambda im: apply_chain(im, drr_chain))


def frame_cost(views, rads, bone, pose, delta, pre, others=(), joint=False, sim=None, per_cam=False):
    """Autoscoper-equivalent cost: sum_cam 1 - NCC(filtered DRR, filtered inverted radiograph) over the
    mask where the filtered DRR of `bone` is non-zero.  pre=(rad_fn, drr_fn).
    joint=True also renders the bones in `others` [(Bone, Pose)] into the DRR (overlap modelled)."""
    rad_fn, drr_fn = pre[:2]
    dil = pre[2] if len(pre) > 2 else None      # None: Autoscoper mask (filtered DRR != 0)
    sim = sim or masked_ncc
    R, t = pose.apply(delta)
    cs = []
    for ci, (v, rad) in enumerate(zip(views, rads)):
        lo, hi = v.roi(bone, R, t)
        raw = v.render([(bone, R, t)], lo, hi)
        own = drr_fn(raw)
        if joint and others:
            drr = drr_fn(v.render([(bone, R, t)] + [(b, p.R, p.t) for b, p in others], lo, hi))
        else:
            drr = own
        if dil is None:
            mask = (own.detach() != 0).to(drr.dtype)
        else:   # silhouette of the unfiltered DRR, dilated by `dil` render pixels
            m = (raw.detach() > 0).float()[None, None]
            mask = (F.max_pool2d(m, 2 * dil + 1, 1, dil)[0, 0] if dil > 0 else m[0, 0]).to(drr.dtype)
        if USE_FOV:
            mask = mask * _fov_mask(ci, v, lo, hi).to(drr.dtype)
        r = rad_fn(v.radiograph(rad, lo, hi)).to(drr.dtype)
        cs.append(1 - sim(drr, r, mask))
    return cs if per_cam else sum(cs)
