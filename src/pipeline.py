"""Shared building blocks for the analysis runners (keyframe paths, per-trial setup, frame-to-frame crops).

methods_comparison/run.py keeps its own copies of these (it produced the methods-comparison results and
is left unchanged); newer analyses (keyframe_gaps/) import them from here.
"""
import numpy as np

from . import collision
from .costs import F2F, fov_masks
from .drr import View
from .geometry import interpolate


def ref_keys(T, keys, bones):
    """Reference (bead-based) poses at the keyframes: {bone: {frame: 4x4}}."""
    return {n: {k: T.ref[n][k] for k in keys} for n in bones}


def spline_track(T, keyposes, kind='spline'):
    """Interpolate keyframe poses over the trial (cubic pivot spline + rotation spline, or linear + slerp).
    Returns {bone: (n_frames, 4, 4)}, NaN outside the first..last keyframe."""
    out = {}
    for n, kp in keyposes.items():
        Ms = np.full((T.n_frames, 4, 4), np.nan); ks = sorted(kp)
        for k in ks: Ms[k] = kp[k]
        out[n] = interpolate(Ms, ks, np.arange(ks[0], ks[-1] + 1), T.bones[n].pivot, kind)
        out[n] = np.concatenate([np.full((ks[0], 4, 4), np.nan), out[n],
                                 np.full((T.n_frames - ks[-1] - 1, 4, 4), np.nan)])
    return out


_setup = {}


def anchored_setup(T, bones):
    """Per-trial objects for the anchored fit, built once per process: FOV masks, views at 256 and 512,
    signed distance fields of the PV volumes."""
    if T.name not in _setup:
        fovs = fov_masks(T)
        views = {r: [View(cam, r, T.sample_distance, T.ray_intensity) for cam in T.cams] for r in (256, 512)}
        sdfs = {n: collision.SDF(T.bones[n]) for n in bones}
        _setup[T.name] = (fovs, views, sdfs)
    return _setup[T.name]


def f2f_for(T, fovs, base, frames, bones, margin=100):
    """Frame-to-frame term over `frames`, with images cropped to where the bones are (+ margin px)."""
    crop = []
    for cam in T.cams:
        uv = np.vstack([cam.project(T.bones[n].corners @ base[n][f][:3, :3].T + base[n][f][:3, 3])
                        for n in bones for f in frames])
        lo = np.clip(np.floor(uv.min(0)) - margin, 0, None).astype(int)
        hi = np.minimum(np.ceil(uv.max(0)) + margin, [cam.W, cam.H]).astype(int)
        crop.append((lo[0], lo[1], hi[0], hi[1]))
    return F2F(T, frames, fovs, crop=crop)
