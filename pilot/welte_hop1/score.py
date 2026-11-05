"""Score every arm in results/ against the reference poses, frame by frame, over the window (frames 20-40).

Per bone and hidden frame (21-29, 31-39): helical rotation (deg), helical translation (mm, along the axis),
pivot distance (mm). Also joint-level helical error (tibia-talus, talus-calcaneus) and minimum surface
separation between neighbouring bones. Writes results/scores.csv, results/summary.md and figures/errors.png.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bvr_opt import collision
from bvr_opt.geometry import helical_error, pivot_error
from bvr_opt.io import Trial

T = Trial(os.path.join(ROOT, 'datasets/Welte2022/data/SOL001A/T0014_hop108_0001'))
RES = os.path.join(HERE, 'results'); FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)
BONES = ['tib', 'tal', 'cal']; NAME = {'tib': 'Tibia', 'tal': 'Talus', 'cal': 'Calcaneus'}
HIDDEN = [f for f in range(21, 40) if f != 30]
ORDER = ['S1_linear', 'S1_spline', 'S1_spline_raw', 'S2', 'S3', 'P1', 'P2', 'P3', 'P3tol', 'P2_drr', 'P2_f2f', 'PF']
LABEL = {'S1_linear': 'S1 slerp interpolation + BW', 'S1_spline': 'S1 spline interpolation + BW',
         'S1_spline_raw': 'S1 spline interpolation, unfiltered', 'S2': 'S2 keyframe simplex + spline + BW',
         'S3': 'S3 per-frame PSO + simplex (512²)', 'P1': 'P1 anchored, bones alone',
         'P2': 'P2 anchored, co-fitted', 'P3': 'P3 = P2 + contact (d_min 0)', 'P3tol': 'P3 + contact (tolerant, d_min −2 mm)',
         'P2_drr': 'P2, DRR term only', 'P2_f2f': 'P2, frame-to-frame only', 'PF': 'Per-frame co-fit (no anchoring)'}


def main():
    sdfs = {n: collision.SDF(T.bones[n]) for n in BONES}
    arms = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in sorted(glob.glob(os.path.join(RES, '*.npz')))}
    parts = [a for a in arms if a.startswith('S3') and a != 'S3']          # S3a, S3b: per-frame PSO jobs
    if parts:
        arms['S3'] = {n: np.nanmean(np.stack([arms[a][n] for a in parts]), 0) for n in BONES}
        for a in parts: arms.pop(a)
    arms = {a: arms[a] for a in ORDER if a in arms} | {a: v for a, v in arms.items() if a not in ORDER}
    rows = []
    for arm, tr in arms.items():
        for f in HIDDEN:
            ok = all(np.isfinite(tr[n][f]).all() for n in BONES)
            for n in BONES:
                M = tr[n][f]
                if not np.isfinite(M).all():
                    continue
                G = T.ref[n][f]; rot, tra = helical_error(M, G)
                rows.append(dict(arm=arm, frame=f, item=n, rot_deg=rot, hel_mm=tra, pivot_mm=pivot_error(M, G, T.bones[n].pivot)))
            if ok:
                for p, d in [('tib', 'tal'), ('tal', 'cal')]:
                    rel = np.linalg.inv(tr[p][f]) @ tr[d][f]; relg = np.linalg.inv(T.ref[p][f]) @ T.ref[d][f]
                    rot, tra = helical_error(rel, relg)
                    P = {k: (torch.tensor(tr[k][f][:3, :3]), torch.tensor(tr[k][f][:3, 3])) for k in (p, d)}
                    rows.append(dict(arm=arm, frame=f, item=f'{p}-{d}', rot_deg=rot, hel_mm=tra,
                                     pivot_mm=pivot_error(rel, relg, T.bones[d].pivot), sep_mm=collision.min_separation(sdfs, P, p, d)))
    for f in HIDDEN:                                                   # reference separations
        P = {k: (torch.tensor(T.ref[k][f][:3, :3]), torch.tensor(T.ref[k][f][:3, 3])) for k in BONES}
        for p, d in [('tib', 'tal'), ('tal', 'cal')]:
            rows.append(dict(arm='reference', frame=f, item=f'{p}-{d}', sep_mm=collision.min_separation(sdfs, P, p, d)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, 'scores.csv'), index=False)

    # ---- summary table: median [max] per bone
    lines = ['| Arm | Frames | ' + ' | '.join(f'{NAME[n]} rot ° | {NAME[n]} pivot mm' for n in BONES) +
             ' | Tal–cal separation mm (ref ' + f"{df[(df.arm == 'reference') & (df.item == 'tal-cal')].sep_mm.median():.2f})" + ' |',
             '|' + '---|' * (3 + 2 * len(BONES))]
    for arm in arms:
        d = df[df.arm == arm]; cells = []
        for n in BONES:
            x = d[d.item == n]
            cells += [f'{x.rot_deg.median():.2f} [{x.rot_deg.max():.2f}]', f'{x.pivot_mm.median():.2f} [{x.pivot_mm.max():.2f}]']
        s = d[d.item == 'tal-cal'].sep_mm
        lines.append(f'| {LABEL.get(arm, arm)} | {d[d.item == "tal"].frame.nunique()} | ' + ' | '.join(cells) +
                     f' | {s.median():.2f} |' if len(s) else f' | – |')
    hel = ['', '| Arm | ' + ' | '.join(f'{NAME[n]} helical translation mm' for n in BONES) + ' | Tib–tal rot ° | Tal–cal rot ° |',
           '|' + '---|' * (3 + len(BONES))]
    for arm in arms:
        d = df[df.arm == arm]
        c = [f'{d[d.item == n].hel_mm.median():.2f} [{d[d.item == n].hel_mm.max():.2f}]' for n in BONES]
        for j in ('tib-tal', 'tal-cal'):
            x = d[d.item == j]; c.append(f'{x.rot_deg.median():.2f} [{x.rot_deg.max():.2f}]' if len(x) else '–')
        hel.append(f'| {LABEL.get(arm, arm)} | ' + ' | '.join(c) + ' |')
    open(os.path.join(RES, 'summary.md'), 'w').write('Median [max] over hidden frames 21–29, 31–39\n\n' + '\n'.join(lines + hel) + '\n')
    print('\n'.join(lines + hel))

    # ---- figure: error by frame
    show = [a for a in ['S1_spline', 'S2', 'S3', 'P1', 'P2', 'P3', 'P3tol', 'PF'] if a in arms]
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for j, n in enumerate(BONES):
        for i, arm in enumerate(show):
            d = df[(df.arm == arm) & (df.item == n)].sort_values('frame')
            kw = dict(marker='o', ms=5, ls='') if arm == 'S3' else dict(marker='.', ms=4)
            ax[0, j].plot(d.frame, d.pivot_mm, color=cols[i], label=LABEL.get(arm, arm), **kw)
            ax[1, j].plot(d.frame, d.rot_deg, color=cols[i], **kw)
        for a in ax[:, j]:
            for k in (20, 30, 40): a.axvline(k, color='0.7', lw=0.8, ls='--')
        ax[0, j].set_title(NAME[n]); ax[1, j].set_xlabel('Frame (dashed: keyframes)')
    ax[0, 0].set_ylabel('Pivot position error (mm)'); ax[1, 0].set_ylabel('Helical rotation error (°)')
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.suptitle('T0014_hop108_0001, frames 20–40: error against the bead-based reference, perfect keyframes every 10 frames')
    plt.tight_layout(rect=(0, 0.08, 1, 0.96)); plt.savefig(os.path.join(FIG, 'errors.png'), dpi=110)


if __name__ == '__main__':
    main()
