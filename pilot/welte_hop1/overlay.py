"""Overlay the Welte reference (bead-based) poses on both cameras for T0014_hop108_0001.

Each bone's DRR is rendered at its reference pose through the MayaCam 2.0 calibration and its
silhouette outline drawn on the radiograph. A matching outline means the camera port, the .tra
convention (column-major) and the volume local frame are all correct.

Usage: python pilot/welte_hop1/overlay.py [frames...]   (default 10 30 50)
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from bvr_opt.drr import View
from bvr_opt.io import Trial

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRIAL = os.path.join(ROOT, 'datasets/Welte2022/data/SOL001A/T0014_hop108_0001')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
COL = {'cal': '#ff4d6d', 'tal': '#4dd2ff', 'tib': '#7dff6a'}
NAME = {'cal': 'Calcaneus', 'tal': 'Talus', 'tib': 'Tibia'}


def main(frames):
    os.makedirs(OUT, exist_ok=True)
    T = Trial(TRIAL)
    views = [View(c, 1024, T.sample_distance, T.ray_intensity) for c in T.cams]
    fig, ax = plt.subplots(2, len(frames), figsize=(5.2 * len(frames), 10.4))
    for j, f in enumerate(frames):
        for ci, cam in enumerate(T.cams):
            a = ax[ci, j]
            im = T.image(ci, f).numpy()
            a.imshow(im, cmap='gray', vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.5))
            los, his = [], []
            for k, b in T.bones.items():
                M = T.ref[k][f]; R, t = torch.tensor(M[:3, :3]), torch.tensor(M[:3, 3])
                lo, hi = views[ci].roi(b, R, t)
                with torch.no_grad():
                    d = views[ci].render([(b, R, t)], lo, hi).numpy()
                rr, cc = [x.numpy() for x in views[ci].pixel_grid(lo, hi)]
                # silhouette outline of the bone's DRR (imshow puts pixel centres at integers: subtract 0.5)
                a.contour(cc - .5, rr - .5, (d > 0.02).astype(float), levels=[0.5], colors=COL[k], linewidths=1.1)
                los.append(lo); his.append(hi)
            lo, hi = np.min(los, 0), np.max(his, 0); pad = 30
            a.set_xlim(lo[0] - pad, hi[0] + pad); a.set_ylim(min(hi[1] + pad, cam.H), max(lo[1] - pad, 0))
            a.set_xticks([]); a.set_yticks([])
            if ci == 0: a.set_title(f'Frame {f} (image {os.path.basename(T.image_files[0][f])[5:9]})', fontsize=11)
            if j == 0: a.set_ylabel(f'Camera {ci + 1}', fontsize=12)
    fig.legend([Line2D([], [], lw=2, color=COL[k]) for k in T.bones], [NAME[k] for k in T.bones],
               loc='lower center', ncol=3, fontsize=11, frameon=False)
    fig.suptitle('T0014_hop108_0001: reference (bead-based) poses, DRR silhouette outlines through the '
                 'MayaCam 2.0 calibration', fontsize=12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    path = os.path.join(OUT, 'reference_overlay.png')
    plt.savefig(path, dpi=110); print(path)


if __name__ == '__main__':
    main([int(x) for x in sys.argv[1:]] or [10, 30, 50])
