"""Point-cloud overlays: CT bone points placed by each method vs by the reference, on both radiographs.

Points are voxels on strong 3D intensity gradients (cortical outline). For each method column, the
reference points are drawn in yellow and the method's points in magenta; where they coincide the
method matches the bead-based pose.

Usage: python pointcloud.py <frame> [zoom bone: tib|tal|cal|all] [arms, comma-separated]
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bvr_opt.geometry import helical_error, pivot_error
from bvr_opt.io import Trial

T = Trial(os.path.join(ROOT, 'datasets/Welte2022/data/SOL001A/T0014_hop108_0001'))
RES = os.path.join(HERE, 'results'); FIG = os.path.join(HERE, 'figures')
COL = {'tib': '#7dff6a', 'tal': '#4dd2ff', 'cal': '#ff5c7a'}
NAME = {'tib': 'tibia', 'tal': 'talus', 'cal': 'calcaneus'}
LABEL = {'S1_spline': 'S1 spline + 10 Hz filter', 'S1_spline_raw': 'S1 spline, unfiltered',
         'S2': 'S2 keyframe simplex + spline + filter', 'S3': 'S3 per-frame PSO', 'P1': 'P1 anchored, bones alone',
         'P2': 'P2 anchored, co-fitted', 'P3': 'P3 co-fitted + contact', 'P3tol': 'P3 contact, tolerant', 'PF': 'Per-frame co-fit'}


def load(arm):
    if arm.startswith('S3'):
        parts = [p for p in os.listdir(RES) if p.startswith('S3') and p.endswith('.npz')]
        tr = [dict(np.load(os.path.join(RES, p))) for p in parts]
        return {n: np.nanmean(np.stack([t[n] for t in tr]), 0) for n in tr[0]}
    return dict(np.load(os.path.join(RES, f'{arm}.npz')))


def main(f, zoom='all', arms=('S1_spline', 'S1_spline_raw', 'S2', 'P1')):
    bones = ['tib', 'tal', 'cal'] if zoom == 'all' else [zoom]
    pts = {n: T.bones[n].surface_points(700 if zoom == 'all' else 900, quantile=0.95, seed=1) for n in bones}
    tracks = {a: load(a) for a in arms}
    fig, ax = plt.subplots(2, len(arms), figsize=(4.6 * len(arms), 9.6 if zoom == 'all' else 8.4))
    for ci, cam in enumerate(T.cams):
        im = T.image(ci, f).numpy()
        ref_uv = {n: cam.project(pts[n] @ T.ref[n][f][:3, :3].T + T.ref[n][f][:3, 3]) for n in bones}
        allr = np.vstack(list(ref_uv.values())); lo, hi = allr.min(0), allr.max(0)
        pad = 40 if zoom == 'all' else 25
        for j, a in enumerate(arms):
            x = ax[ci, j]
            x.imshow(im, cmap='gray', vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5))
            title = [LABEL.get(a, a)]
            for n in bones:
                M = tracks[a][n][f]
                x.scatter(*(ref_uv[n] - .5).T, s=5 if zoom != 'all' else 3, c='#ffd400', alpha=.9, lw=0)
                if np.isfinite(M).all():
                    uv = cam.project(pts[n] @ M[:3, :3].T + M[:3, 3])
                    x.scatter(*(uv - .5).T, s=5 if zoom != 'all' else 3, c='#ff2bd6', alpha=.9, lw=0)
                    if ci == 0:
                        title.append(f'{NAME[n]}: {pivot_error(M, T.ref[n][f], T.bones[n].pivot):.2f} mm, '
                                     f'{helical_error(M, T.ref[n][f])[0]:.2f}°')
            x.set_xlim(lo[0] - pad, hi[0] + pad); x.set_ylim(min(hi[1] + pad, cam.H), max(lo[1] - pad, 0))
            x.set_xticks([]); x.set_yticks([])
            if ci == 0: x.set_title('\n'.join(title), fontsize=9)
            if j == 0: x.set_ylabel(f'Camera {ci + 1}', fontsize=11)
    what = 'tibia, talus, calcaneus' if zoom == 'all' else NAME[zoom]
    fig.suptitle(f'Frame {f} (hidden; keyframes at 30 and 40): {what}. yellow = reference (bead-based) pose, '
                 f'magenta = method pose', fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(FIG, f'pointcloud_f{f}_{zoom}.png'); plt.savefig(path, dpi=110); print(path)


if __name__ == '__main__':
    f = int(sys.argv[1]); zoom = sys.argv[2] if len(sys.argv) > 2 else 'all'
    arms = sys.argv[3].split(',') if len(sys.argv) > 3 else ('S1_spline', 'S1_spline_raw', 'S2', 'P1')
    main(f, zoom, arms)
