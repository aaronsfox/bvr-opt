"""Score every assembled methods-comparison track against the bead-based reference, frame by frame.

    python methods_comparison/score.py            # after `run.py assemble`

Scored frames: strictly between the first and last keyframe of each (trial, offset), keyframes excluded.
Per bone: helical rotation (deg), helical translation along the axis (mm), pivot position error (mm).
Per joint (tib-tal, tal-cal): the same for the distal bone relative to the proximal one, and the minimum
surface separation (PV volumes, signed distance field). Every arm is scored unfiltered (primary) and
after the same dual-pass Butterworth filter (secondary). A frame "needs correction" when pivot error
> 1 mm or helical rotation > 2 deg (Welte 2022 operator standard).

Writes $BVR_RESULTS/scores.csv, compute.csv, summary.md and figures/. The summary ends with a compute table
(renders and wall time per bone-frame for each arm, read from the unit files).
"""
import glob
import json
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
from src import plotstyle as ps  # noqa: E402
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
    """Error distribution per arm and bone (unfiltered); each arm keeps its palette colour and marker."""
    ps.apply()
    d0 = df[df['filter'] == 'none']; arms = list(dict.fromkeys(d0.arm))
    fig, ax = plt.subplots(2, 3, figsize=(ps.DOUBLE, 4.6), squeeze=False, sharex=True)
    for j, n in enumerate(BONES):
        for i, m in enumerate(['pivot_mm', 'rot_deg']):
            a = ax[i, j]
            data = [d0[(d0.arm == arm) & (d0.item == n)][m].values for arm in arms]
            bp = a.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color='black', lw=1.2),
                           whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
            for k, (box, fl) in enumerate(zip(bp['boxes'], bp['fliers'])):
                box.set(facecolor=ps.color(k), alpha=0.75, lw=0.8)
                fl.set(marker=ps.marker(k), markerfacecolor=ps.color(k), markeredgecolor=ps.color(k), markersize=2, alpha=0.6)
            a.set_yscale('log'); ps.plain_log(a); a.axhline(TH['pivot_mm'] if m == 'pivot_mm' else TH['rot_deg'], color='0.5', ls='--', lw=0.8)
            a.set_xticks(range(1, len(arms) + 1), arms, rotation=45, ha='right')
            if j == 0: a.set_ylabel('Pivot error (mm)' if m == 'pivot_mm' else 'Helical rotation error (°)')
        ax[0, j].set_title(NAME[n])
    fig.tight_layout(); ps.save(fig, path); plt.close(fig)


def compute_table(arms):
    """Compute cost per arm from the unit files (implementation-independent counts first, then wall time).

    Renders per bone-frame = DRR renders of one bone in both views needed per scored bone-frame:
      pso_simplex arms: Autoscoper cost evaluations (one render each);
      new_opt arms: iterations x frames sampled per gap / frames in the gap (co-fitted bones share a render
        pass, each with gradients); new_opt_per_frame: iterations per frame; interpolation: 0.
    Wall time per bone-frame = unit time / bone-frames it produced, on the recorded device. Several worker
    processes may share one GPU (column `workers`), which inflates wall time relative to a dedicated GPU."""
    rows = []
    for arm in arms:
        for p in glob.glob(os.path.join(RESULTS, arm, 'units', '*.json')):
            d = json.load(open(p)); c = d['config']
            n_bf = sum(len(v) for v in d['poses'].values())
            if n_bf == 0:
                continue
            if c['kind'] == 'autoscoper':
                renders = sum(m['evals'] for m in d['meta'].values()) / n_bf
            elif c['kind'] == 'anchored':
                inner = d['unit']['b'] - d['unit']['a'] - 1
                renders = sum(c['iters']) * min(c['batch'], inner) / inner
            elif c['kind'] == 'perframe':
                renders = sum(c['iters'])
            else:
                renders = 0.0
            rows.append(dict(arm=arm, bone_frames=n_bf, renders=renders, sec=d['sec'], sec_per_bf=d['sec'] / n_bf,
                             device=d.get('gpu') or d.get('device'), workers=d.get('workers', 1)))
    if not rows:
        return '', None
    cdf = pd.DataFrame(rows)
    out = ['## Compute', '',
           'Renders per bone-frame: DRR renders of one bone (both views) per scored bone-frame; with gradients for '
           'new_opt arms. Wall time on the recorded device, with `workers` processes sharing it.', '',
           '| Arm | Bone-frames | Renders per bone-frame | Wall time per bone-frame (s) | Total compute (device-hours) | Device | Workers |',
           '|---|---|---|---|---|---|---|']
    for arm, g in cdf.groupby('arm', sort=False):
        out.append(f'| {arm} | {g.bone_frames.sum()} | {(g.renders * g.bone_frames).sum() / g.bone_frames.sum():.0f} | '
                   f'{g.sec.sum() / g.bone_frames.sum():.2f} | {g.sec.sum() / 3600:.2f} | {g.device.iloc[0]} | '
                   f'{int(g.workers.max())} |')
    return '\n'.join(out) + '\n', cdf


def main():
    sdfs = {tn: {n: collision.SDF(trial(tn).bones[n]) for n in BONES} for tn in CFG['trials']}
    rows = []
    for arm in CFG['arms']:
        for p in sorted(glob.glob(os.path.join(RESULTS, arm, 'tracks', '*.npz'))):
            tn, o = os.path.basename(p)[:-4].rsplit('_o', 1)
            rows += score_track(arm, tn, int(o), dict(np.load(p)), sdfs)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'scores.csv'), index=False)
    comp, cdf = compute_table([a for a in CFG['arms']])
    if cdf is not None:
        cdf.to_csv(os.path.join(RESULTS, 'compute.csv'), index=False)
    s = summary(df) + '\n' + comp; open(os.path.join(RESULTS, 'summary.md'), 'w').write(s); print(s)
    os.makedirs(os.path.join(RESULTS, 'figures'), exist_ok=True)
    figure(df, os.path.join(RESULTS, 'figures', 'errors_by_arm.png'))


if __name__ == '__main__':
    main()
