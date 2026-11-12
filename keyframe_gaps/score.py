"""Score every assembled keyframe-gaps track against the bead-based reference, frame by frame.

    python keyframe_gaps/score.py          # after `run.py assemble`

Scored frames: non-keyframes strictly between the bookends. Per bone: pivot error (mm), helical rotation
(deg), helical translation (mm); per joint (tib-tal, tal-cal): relative rotation/translation error and
minimum surface separation (PV signed distance field). Each scored frame also carries the length of its
gap and its distance to the nearest keyframe. A frame "needs correction" when pivot error > 1 mm or
helical rotation > 2 deg (Welte 2022 operator standard). Results are unfiltered.

Two frame sets are summarised: all scored frames of each spacing (primary) and the frames that are not a
keyframe at any spacing (common set, so every spacing is scored on identical frames).

Arms with keyframe correction also get a table of their corrected keyframes' error (keyframes.csv).

Writes $BVR_RESULTS/scores.csv, compute.csv, keyframes.csv, summary.md and figures/.
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
from run import CFG, RESULTS, arm_dir, keyframes, trial  # noqa: E402
from src import collision  # noqa: E402
from src import plotstyle as ps  # noqa: E402
from src.geometry import helical_error, pivot_error  # noqa: E402

BONES = CFG['bones']; JOINTS = [('tib', 'tal'), ('tal', 'cal')]
NAME = {'tib': 'Tibia', 'tal': 'Talus', 'cal': 'Calcaneus'}
TH = CFG['correction_threshold']
SP = [str(s) for s in CFG['spacings']]
# arms to score: configured arms, plus the chains-only path of stepwise arms (written by run.py assemble)
ARMS = [x for a, c in CFG['arms'].items() for x in ([a, a + '_chains'] if c['kind'] == 'stepwise' else [a])]


def needs_correction(d):
    return (d.pivot_mm > TH['pivot_mm']) | (d.rot_deg > TH['rot_deg'])


def common_frames(tn):
    N = trial(tn).n_frames; keys = set().union(*[keyframes(N, s) for s in SP])
    return {f for f in range(CFG['bookend'] + 1, N - 1 - CFG['bookend']) if f not in keys}


def score_track(arm, s, tn, tr, sdfs):
    T = trial(tn); keys = keyframes(T.n_frames, s); common = common_frames(tn)
    rows = []
    for a, b in zip(keys[:-1], keys[1:]):
        for f in range(a + 1, b):
            info = dict(arm=arm, spacing=s, trial=tn, frame=f, gap=b - a, to_key=min(f - a, b - f), common=f in common)
            for n in BONES:
                M, G = tr[n][f], T.ref[n][f]
                if not np.isfinite(M).all():
                    continue
                rot, hel = helical_error(M, G)
                rows.append(dict(info, item=n, rot_deg=rot, hel_mm=hel, pivot_mm=pivot_error(M, G, T.bones[n].pivot)))
            for p, d in JOINTS:
                if not (np.isfinite(tr[p][f]).all() and np.isfinite(tr[d][f]).all()):
                    continue
                rel = np.linalg.inv(tr[p][f]) @ tr[d][f]; relg = np.linalg.inv(T.ref[p][f]) @ T.ref[d][f]
                rot, hel = helical_error(rel, relg)
                P = {k: (torch.tensor(tr[k][f][:3, :3]), torch.tensor(tr[k][f][:3, 3])) for k in (p, d)}
                rows.append(dict(info, item=f'{p}-{d}', rot_deg=rot, hel_mm=hel,
                                 pivot_mm=pivot_error(rel, relg, T.bones[d].pivot), sep_mm=collision.min_separation(sdfs, P, p, d)))
    return rows


def q95(x):
    return np.nanpercentile(x, 95) if len(x) else np.nan


def table(df, title, note):
    out = [f'## {title}', '', note, '',
           '| Arm | Spacing | Bone-frames | ' + ' | '.join(f'{NAME[n]} pivot mm | {NAME[n]} rot °' for n in BONES) +
           ' | Max pivot mm | Tal–cal rot ° | Needs correction % |', '|' + '---|' * (3 + 2 * len(BONES) + 3)]
    for arm in ARMS:
        for s in SP:
            d = df[(df.arm == arm) & (df.spacing == s)]
            if d.empty:
                continue
            b = d[d.item.isin(BONES)]; cells = []
            for n in BONES:
                x = d[d.item == n]
                cells += [f'{x.pivot_mm.median():.2f} [{q95(x.pivot_mm):.2f}]', f'{x.rot_deg.median():.2f} [{q95(x.rot_deg):.2f}]']
            j = d[d.item == 'tal-cal']
            out.append(f'| {arm} | {s} | {len(b)} | ' + ' | '.join(cells) + f' | {b.pivot_mm.max():.2f} | '
                       f'{j.rot_deg.median():.2f} [{q95(j.rot_deg):.2f}] | {100 * needs_correction(b).mean():.1f} |')
    return out + ['']


def by_trial(df):
    out = ['## Frames needing correction by trial (% of bone-frames; gap lengths used in brackets)', '',
           '| Arm | Spacing | ' + ' | '.join(CFG['trials']) + ' |', '|' + '---|' * (2 + len(CFG['trials']))]
    for arm in ARMS:
        for s in SP:
            d = df[(df.arm == arm) & (df.spacing == s) & df.item.isin(BONES)]
            if d.empty:
                continue
            c = []
            for tn in CFG['trials']:
                x = d[d.trial == tn]
                gaps = sorted(set(np.diff(keyframes(trial(tn).n_frames, s))))
                c.append(f'{100 * needs_correction(x).mean():.1f} ({",".join(map(str, gaps))})' if len(x) else '–')
            out.append(f'| {arm} | {s} | ' + ' | '.join(c) + ' |')
    return out + ['']


def by_distance(df):
    """Median pivot error by distance to the nearest keyframe (bones pooled)."""
    b = df[df.item.isin(BONES)].copy()
    b['to_key_bin'] = pd.cut(b.to_key, [0, 2, 5, 10, 15, 20, 100], labels=['1–2', '3–5', '6–10', '11–15', '16–20', '>20'])
    t = b.pivot_table(index=['arm', 'spacing'], columns='to_key_bin', values='pivot_mm', aggfunc='median', observed=False)
    out = ['## Median pivot error (mm) by distance to the nearest keyframe (frames)', '',
           '| Arm | Spacing | ' + ' | '.join(map(str, t.columns)) + ' |', '|' + '---|' * (2 + len(t.columns))]
    for arm in ARMS:
        for s in SP:
            if (arm, s) in t.index:
                out.append(f'| {arm} | {s} | ' + ' | '.join('–' if np.isnan(v) else f'{v:.2f}' for v in t.loc[(arm, s)]) + ' |')
    return out + ['']


def keyframe_table():
    """Keyframe correction: error of the given and the corrected keyframes against the reference."""
    rows = []
    for arm, c in CFG['arms'].items():
        if 'keyframe_correction' not in c:
            continue
        for s in SP:
            for p in glob.glob(os.path.join(arm_dir(arm, s), 'keyframes', '*.json')):
                d = json.load(open(p)); T = trial(d['unit']['trial'])
                for n, kp in d['keyframes'].items():
                    for k, M in kp.items():
                        k = int(k); M = np.array(M); G = T.ref[n][k]
                        rows.append(dict(arm=arm, spacing=s, trial=T.name, frame=k, item=n,
                                         pivot_mm=pivot_error(M, G, T.bones[n].pivot), rot_deg=helical_error(M, G)[0]))
    if not rows:
        return [], None
    k = pd.DataFrame(rows)
    out = ['## Keyframe correction', '',
           f"Error of the corrected keyframes against the reference (given keyframes: {CFG['keyframes']}). "
           'Median [95th percentile] and maximum, bones pooled.', '',
           '| Arm | Spacing | Keyframes | Pivot mm | Max pivot mm | Rotation ° | Max rotation ° |', '|---|---|---|---|---|---|---|']
    for (arm, s), g in k.groupby(['arm', 'spacing'], sort=False):
        out.append(f'| {arm} | {s} | {len(g)} | {g.pivot_mm.median():.2f} [{q95(g.pivot_mm):.2f}] | {g.pivot_mm.max():.2f} | '
                   f'{g.rot_deg.median():.2f} [{q95(g.rot_deg):.2f}] | {g.rot_deg.max():.2f} |')
    return out + [''], k


def compute(arms):
    rows = []
    for arm in arms:
        for s in SP:
            for p in glob.glob(os.path.join(arm_dir(arm, s), 'units', '*.json')):
                d = json.load(open(p)); n_bf = sum(len(v) for v in d['poses'].values())
                m = d['meta']
                renders = sum(m['iters']) * m['batch'] / (m['gap'] - 1) if 'gap' in m else 0.0
                rows.append(dict(arm=arm, spacing=s, bone_frames=n_bf, renders=renders, sec=d['sec'],
                                 device=d.get('gpu') or d.get('device'), workers=d.get('workers', 1)))
    if not rows:
        return [], None
    c = pd.DataFrame(rows)
    out = ['## Compute', '', 'Renders per bone-frame: DRR renders of one bone (both views, with gradients) per scored '
           'bone-frame. Wall time on the recorded device, with `workers` processes sharing it.', '',
           '| Arm | Spacing | Bone-frames | Renders per bone-frame | Wall time per bone-frame (s) | Device-hours | Device | Workers |',
           '|---|---|---|---|---|---|---|---|']
    for (arm, s), g in c.groupby(['arm', 'spacing'], sort=False):
        out.append(f'| {arm} | {s} | {g.bone_frames.sum()} | {(g.renders * g.bone_frames).sum() / g.bone_frames.sum():.0f} | '
                   f'{g.sec.sum() / g.bone_frames.sum():.2f} | {g.sec.sum() / 3600:.2f} | {g.device.iloc[0]} | {int(g.workers.max())} |')
    return out + [''], c


def figure(df, path):
    """Pivot error against keyframe spacing per bone (perfect keyframes): median solid, 95th percentile dashed."""
    ps.apply()
    arms = [a for a in ARMS if (df.arm == a).any()]
    fig, ax = plt.subplots(1, 3, figsize=(ps.DOUBLE, 2.6), sharey=True)
    x = np.arange(len(SP))
    for i, arm in enumerate(arms):
        for j, n in enumerate(BONES):
            d = df[(df.arm == arm) & (df.item == n)]
            med = [d[d.spacing == s].pivot_mm.median() for s in SP]; p95 = [q95(d[d.spacing == s].pivot_mm) for s in SP]
            ax[j].plot(x, med, '-', **ps.style(i), label=f'{arm}, median' if j == 0 else None)
            ax[j].plot(x, p95, '--', **ps.style(i, hollow=True), label=f'{arm}, 95th percentile' if j == 0 else None)
    for j, n in enumerate(BONES):
        ax[j].axhline(TH['pivot_mm'], color='0.5', ls=':', lw=0.8); ax[j].set_xticks(x, SP)
        ax[j].set_title(NAME[n]); ax[j].set_xlabel('Keyframe spacing (frames)'); ax[j].set_yscale('log'); ps.plain_log(ax[j])
    ax[0].set_ylabel('Pivot error (mm)')
    fig.legend(loc='lower center', ncol=min(4, 2 * len(arms)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06 * (1 + (2 * len(arms) - 1) // 4), 1, 1)); ps.save(fig, path); plt.close(fig)


def main():
    sdfs = {tn: {n: collision.SDF(trial(tn).bones[n]) for n in BONES} for tn in CFG['trials']}
    rows = []
    for arm in ARMS:
        for s in SP:
            for p in sorted(glob.glob(os.path.join(arm_dir(arm, s), 'tracks', '*.npz'))):
                tn = os.path.basename(p)[:-4]; rows += score_track(arm, s, tn, dict(np.load(p)), sdfs[tn])
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, 'scores.csv'), index=False)
    comp, cdf = compute(list(CFG['arms']))
    if cdf is not None: cdf.to_csv(os.path.join(RESULTS, 'compute.csv'), index=False)
    lines = ['# Keyframe gaps results', '',
             'Perfect keyframes; bookends at frame 5 and N − 6. Median [95th percentile], unfiltered, pooled over trials.', '']
    lines += table(df, 'All scored frames per spacing (primary)', 'Every non-keyframe between the bookends, for each spacing.')
    lines += table(df[df.common], 'Common frame set', 'Frames that are not a keyframe at any spacing, so every spacing is scored on identical frames.')
    kt, kdf = keyframe_table()
    if kdf is not None: kdf.to_csv(os.path.join(RESULTS, 'keyframes.csv'), index=False)
    lines += by_trial(df) + by_distance(df) + kt + comp
    s = '\n'.join(lines) + '\n'; open(os.path.join(RESULTS, 'summary.md'), 'w').write(s); print(s)
    os.makedirs(os.path.join(RESULTS, 'figures'), exist_ok=True)
    figure(df, os.path.join(RESULTS, 'figures', 'error_by_spacing.png'))


if __name__ == '__main__':
    main()
