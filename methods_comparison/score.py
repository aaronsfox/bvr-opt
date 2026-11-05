"""Score every assembled methods-comparison track against the bead-based reference, frame by frame.

    python methods_comparison/score.py            # after `run.py assemble`

Scored frames: strictly between the first and last keyframe of each (trial, offset), keyframes excluded.
Per bone: helical rotation (deg), helical translation along the axis (mm), pivot position error (mm).
Per joint (tib-tal, tal-cal): the same for the distal bone relative to the proximal one, and the minimum
surface separation (PV volumes, signed distance field). Every arm is scored unfiltered (primary) and
after the same dual-pass Butterworth filter (secondary). A frame "needs correction" when pivot error
> 1 mm or helical rotation > 2 deg (Welte 2022 operator standard).

Writes $BVR_RESULTS/scores.csv, summary.md and figures/.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import CFG, RESULTS, keyframes, trial  # noqa: E402
from src import collision  # noqa: E402
from src.geometry import butterworth, helical_error, pivot_error  # noqa: E402

BONES = CFG['bones']; JOINTS = [('tib', 'tal'), ('tal', 'cal')]
NAME = {'tib': 'Tibia', 'tal': 'Talus', 'cal': 'Calcaneus', 'tib-tal': 'Tibia–talus', 'tal-cal': 'Talus–calcaneus'}
TH = CFG['correction_threshold']


def filtered(T, tr, keys):
    out = {}
    for n in BONES:
        M = tr[n].copy(); seg = slice(keys[0], keys[-1] + 1)
        if np.isfinite(M[seg]).all():
            M[seg] = butterworth(M[seg], T.bones[n].pivot, **CFG['butterworth'])
        else:
            M[seg] = np.nan
        out[n] = M
    return out


def score_track(arm, tn, o, tr, sdfs):
    T = trial(tn); keys = keyframes(T.n_frames, o)
    frames = [f for f in range(keys[0] + 1, keys[-1]) if f not in keys]
    rows = []
    for filt, X in (('none', tr), ('bw', filtered(T, tr, keys))):
        for f in frames:
            for n in BONES:
                M, G = X[n][f], T.ref[n][f]
                if not np.isfinite(M).all():
                    continue
                rot, hel = helical_error(M, G)
                rows.append(dict(arm=arm, trial=tn, offset=o, filter=filt, frame=f, item=n, rot_deg=rot, hel_mm=hel,
                                 pivot_mm=pivot_error(M, G, T.bones[n].pivot)))
            for p, d in JOINTS:
                if not (np.isfinite(X[p][f]).all() and np.isfinite(X[d][f]).all()):
                    continue
                rel = np.linalg.inv(X[p][f]) @ X[d][f]; relg = np.linalg.inv(T.ref[p][f]) @ T.ref[d][f]
                rot, hel = helical_error(rel, relg)
                P = {k: (torch.tensor(X[k][f][:3, :3]), torch.tensor(X[k][f][:3, 3])) for k in (p, d)}
                rows.append(dict(arm=arm, trial=tn, offset=o, filter=filt, frame=f, item=f'{p}-{d}', rot_deg=rot,
                                 hel_mm=hel, pivot_mm=pivot_error(rel, relg, T.bones[d].pivot),
                                 sep_mm=collision.min_separation(sdfs[tn], P, p, d)))
    return rows


def q95(x):
    return np.nanpercentile(x, 95) if len(x) else np.nan


def summary(df):
    arms = list(dict.fromkeys(df.arm))
    out = ['# Methods comparison results', '',
           'Pooled over trials and keyframe offsets; scored frames between keyframes only. '
           'Median [95th percentile].', '']
    for filt, title in (('none', 'Primary: unfiltered'), ('bw', 'Secondary: Butterworth 10 Hz on every arm')):
        d0 = df[df['filter'] == filt]
        out += [f'## {title}', '', '| Arm | Frames | ' + ' | '.join(f'{NAME[n]} pivot mm | {NAME[n]} rot °' for n in BONES)
                + ' | Tal–cal rot ° | Needs correction % |', '|' + '---|' * (3 + 2 * len(BONES) + 1)]
        for arm in arms:
            d = d0[d0.arm == arm]
            if d.empty:
                continue
            cells = []
            for n in BONES:
                x = d[d.item == n]
                cells += [f'{x.pivot_mm.median():.2f} [{q95(x.pivot_mm):.2f}]', f'{x.rot_deg.median():.2f} [{q95(x.rot_deg):.2f}]']
            j = d[d.item == 'tal-cal']
            b = d[d.item.isin(BONES)]
            corr = 100 * ((b.pivot_mm > TH['pivot_mm']) | (b.rot_deg > TH['rot_deg'])).mean()
            out.append(f'| {arm} | {d[d.item == BONES[0]].shape[0]} | ' + ' | '.join(cells) +
                       f' | {j.rot_deg.median():.2f} [{q95(j.rot_deg):.2f}] | {corr:.1f} |')
        out.append('')
    d0 = df[(df['filter'] == 'none') & df.item.isin(BONES)]
    out += ['## Frames needing correction by trial (unfiltered, % of bone-frames)', '',
            '| Arm | ' + ' | '.join(CFG['trials']) + ' |', '|' + '---|' * (1 + len(CFG['trials']))]
    for arm in arms:
        c = []
        for tn in CFG['trials']:
            b = d0[(d0.arm == arm) & (d0.trial == tn)]
            c.append(f'{100 * ((b.pivot_mm > TH["pivot_mm"]) | (b.rot_deg > TH["rot_deg"])).mean():.1f}' if len(b) else '–')
        out.append(f'| {arm} | ' + ' | '.join(c) + ' |')
    return '\n'.join(out) + '\n'


def figure(df, path):
    d0 = df[df['filter'] == 'none']; arms = list(dict.fromkeys(d0.arm))
    fig, ax = plt.subplots(2, 3, figsize=(4 + 1.1 * len(arms) * 3, 8), squeeze=False)
    for j, n in enumerate(BONES):
        for i, m in enumerate(['pivot_mm', 'rot_deg']):
            data = [d0[(d0.arm == a) & (d0.item == n)][m].values for a in arms]
            ax[i, j].boxplot(data, showfliers=True, flierprops=dict(ms=2))
            ax[i, j].set_xticks(range(1, len(arms) + 1), arms, rotation=45, ha='right')
            ax[i, j].set_yscale('log'); ax[i, j].axhline(TH['pivot_mm'] if m == 'pivot_mm' else TH['rot_deg'], color='0.6', ls='--', lw=0.8)
            if j == 0: ax[i, j].set_ylabel('Pivot error (mm)' if m == 'pivot_mm' else 'Helical rotation error (°)')
        ax[0, j].set_title(NAME[n])
    fig.suptitle('Methods comparison: error against the bead-based reference (unfiltered; dashed: Welte operator standard)')
    plt.tight_layout(); plt.savefig(path, dpi=110)


def main():
    sdfs = {tn: {n: collision.SDF(trial(tn).bones[n]) for n in BONES} for tn in CFG['trials']}
    rows = []
    for arm in CFG['arms']:
        for p in sorted(glob.glob(os.path.join(RESULTS, arm, 'tracks', '*.npz'))):
            tn, o = os.path.basename(p)[:-4].rsplit('_o', 1)
            rows += score_track(arm, tn, int(o), dict(np.load(p)), sdfs)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'scores.csv'), index=False)
    s = summary(df); open(os.path.join(RESULTS, 'summary.md'), 'w').write(s); print(s)
    os.makedirs(os.path.join(RESULTS, 'figures'), exist_ok=True)
    figure(df, os.path.join(RESULTS, 'figures', 'errors_by_arm.png'))


if __name__ == '__main__':
    main()
