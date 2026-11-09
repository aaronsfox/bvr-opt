"""Score the rater runs against the bead-based reference and compare with the ideal-keyframe results.

    python rater_gaps/score.py [--interp] [--assemble]

--interp     run spline interpolation through every rater's keyframes first (CPU, seconds per rater)
--assemble   assemble each rater's units into tracks first

For every level, rater, spacing and arm, frames between the bookends that are not keyframes are scored:
pivot error (mm), helical rotation (deg) and helical translation (mm, Welte's metric) per bone, and the
talus-calcaneus relative rotation. A frame needs correction under two criteria: Welte's (helical rotation
> 2 deg or helical translation > 1 mm) and the stricter positional one used so far (pivot error > 1 mm or
rotation > 2 deg). The keyframes themselves are scored as given by the rater and after keyframe correction.
Ideal-keyframe rows come from results/keyframe_gaps/scores.csv.

Writes results/rater_gaps/scores.csv, keyframes.csv, summary.md and figures/.
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'keyframe_gaps')); sys.path.insert(1, ROOT)
import run as kg  # noqa: E402  (keyframe_gaps/run.py)
from src.geometry import helical_error, pivot_error  # noqa: E402

RG = yaml.safe_load(open(os.path.join(ROOT, 'rater_gaps', 'config.yaml')))
OUT = os.environ.get('RG_RESULTS', os.path.join(ROOT, 'results', 'rater_gaps'))
BONES = kg.BONES; SP = [str(s) for s in RG['spacings']]
ARMS = [RG['reference_arm'], RG['arm']]


def rater_dirs():
    for L in RG['levels']:
        for i in range(RG['raters']):
            yield L, f'r{i:02d}', os.path.join(OUT, f'L{L}', f'r{i:02d}')


def use_rater(L, r, path):
    kg.RESULTS = path; kg.SOURCE['keyframes'] = f'rater:L{L}/{r}'
    kg.SOURCE['kc_tol'] = RG['kc_tol'][L]


def needs(d):
    return dict(welte=(d.rot_deg > 2) | (d.hel_mm > 1), pivot=(d.rot_deg > 2) | (d.pivot_mm > 1))


def score_tracks():
    rows, krows = [], []
    for L, r, path in rater_dirs():
        use_rater(L, r, path)
        for arm in ARMS:
            for s in SP:
                for p in sorted(glob.glob(os.path.join(kg.arm_dir(arm, s), 'tracks', '*.npz'))):
                    tn = os.path.basename(p)[:-4]; T = kg.trial(tn); tr = dict(np.load(p))
                    keys = kg.keyframes(T.n_frames, s)
                    for f in range(keys[0] + 1, keys[-1]):
                        if f in keys:
                            continue
                        base = dict(level=str(L), rater=r, arm=arm, spacing=s, trial=tn, frame=f)
                        for n in BONES:
                            M, G = tr[n][f], T.ref[n][f]
                            if not np.isfinite(M).all():
                                continue
                            rot, hel = helical_error(M, G)
                            rows.append(dict(base, item=n, rot_deg=rot, hel_mm=hel, pivot_mm=pivot_error(M, G, T.bones[n].pivot)))
                        if np.isfinite(tr['tal'][f]).all() and np.isfinite(tr['cal'][f]).all():
                            rel = np.linalg.inv(tr['tal'][f]) @ tr['cal'][f]; relg = np.linalg.inv(T.ref['tal'][f]) @ T.ref['cal'][f]
                            rows.append(dict(base, item='tal-cal', rot_deg=helical_error(rel, relg)[0]))
        # keyframes: as given by the rater, and after correction
        for s in SP:
            for tn in kg.CFG['trials']:
                T = kg.trial(tn); keys = kg.keyframes(T.n_frames, s)
                given = kg.given_keyframes(T, keys)
                cp = kg.unit_path(dict(arm=RG['arm'], spacing=s, trial=tn, stage='keys'))
                corr = json.load(open(cp))['keyframes'] if os.path.exists(cp) else None
                for n in BONES:
                    for k in keys:
                        G = T.ref[n][k]
                        for kind, M in (('given', given[n][k]), ('corrected', np.array(corr[n][str(k)]) if corr else None)):
                            if M is None:
                                continue
                            rot, hel = helical_error(M, G)
                            krows.append(dict(level=str(L), rater=r, spacing=s, trial=tn, frame=k, item=n, kind=kind,
                                              rot_deg=rot, hel_mm=hel, pivot_mm=pivot_error(M, G, T.bones[n].pivot)))
    return pd.DataFrame(rows), pd.DataFrame(krows)


def ideal_rows():
    p = os.path.join(ROOT, RG['ideal_results'], 'scores.csv')
    if not os.path.exists(p):
        return pd.DataFrame()
    d = pd.read_csv(p, dtype={'spacing': str})
    d = d[d.arm.isin(RG['ideal_arms']) & d.item.isin(BONES + ['tal-cal'])].copy()
    d['level'] = 'ideal'; d['rater'] = '-'
    return d[['level', 'rater', 'arm', 'spacing', 'trial', 'frame', 'item', 'rot_deg', 'hel_mm', 'pivot_mm']]


def q95(x):
    return np.nanpercentile(x, 95) if len(x) else np.nan


def main_table(df):
    out = ['## Between-keyframe error by rater level and spacing', '',
           'Bones pooled; median [95th percentile]. Needs correction: % of bone-frames beyond Welte\'s standard '
           '(helical rotation > 2° or helical translation > 1 mm) and beyond the positional criterion (pivot > 1 mm '
           'or rotation > 2°). "ideal" = reference keyframes (perfect rater).', '',
           '| Keyframes | Arm | Spacing | Bone-frames | Pivot mm | Rotation ° | Helical translation mm | Tal–cal rot ° | Needs correction, Welte % | Needs correction, pivot % |',
           '|---|---|---|---|---|---|---|---|---|---|']
    order = ['ideal'] + [str(L) for L in RG['levels']]
    for s in SP:
        for lv in order:
            for arm in (RG['ideal_arms'] if lv == 'ideal' else ARMS):
                d = df[(df.level == lv) & (df.arm == arm) & (df.spacing == s)]
                b = d[d.item.isin(BONES)]
                if b.empty:
                    continue
                j = d[d.item == 'tal-cal']; nc = needs(b)
                out.append(f'| {"ideal" if lv == "ideal" else lv + "×"} | {arm} | {s} | {len(b)} | '
                           f'{b.pivot_mm.median():.2f} [{q95(b.pivot_mm):.2f}] | {b.rot_deg.median():.2f} [{q95(b.rot_deg):.2f}] | '
                           f'{b.hel_mm.median():.2f} [{q95(b.hel_mm):.2f}] | {j.rot_deg.median():.2f} [{q95(j.rot_deg):.2f}] | '
                           f'{100 * nc["welte"].mean():.1f} | {100 * nc["pivot"].mean():.1f} |')
    return out + ['']


def bone_table(df):
    out = ['## Median pivot error (mm) / rotation (°) per bone, stepwise method', '',
           '| Keyframes | Spacing | ' + ' | '.join(kg.CFG.get('bone_names', {}).get(n, n) for n in BONES) + ' |',
           '|---|---|' + '---|' * len(BONES)]
    for s in SP:
        for lv, arm in [('ideal', 'new_opt_stepwise')] + [(str(L), RG['arm']) for L in RG['levels']]:
            d = df[(df.level == lv) & (df.arm == arm) & (df.spacing == s)]
            if d.empty:
                continue
            out.append(f'| {"ideal" if lv == "ideal" else lv + "×"} | {s} | ' +
                       ' | '.join(f'{d[d.item == n].pivot_mm.median():.2f} / {d[d.item == n].rot_deg.median():.2f}' for n in BONES) + ' |')
    return out + ['']


def keyframe_table(k):
    if k.empty:
        return []
    out = ['## Keyframes: as given by the rater vs after keyframe correction', '',
           'Bones pooled; median [95th percentile]. Improved: % of bone-keyframes whose pivot error decreased.', '',
           '| Level | Spacing | Keyframes | Given pivot mm | Corrected pivot mm | Given rotation ° | Corrected rotation ° | Improved % |',
           '|---|---|---|---|---|---|---|---|']
    for L in RG['levels']:
        for s in SP:
            g = k[(k.level == str(L)) & (k.spacing == s) & (k.kind == 'given')]
            c = k[(k.level == str(L)) & (k.spacing == s) & (k.kind == 'corrected')]
            if g.empty:
                continue
            m = g.merge(c, on=['rater', 'trial', 'frame', 'item'], suffixes=('_g', '_c'))
            imp = f'{100 * (m.pivot_mm_c < m.pivot_mm_g).mean():.0f}' if len(m) else '–'
            cc = (f'{c.pivot_mm.median():.2f} [{q95(c.pivot_mm):.2f}]', f'{c.rot_deg.median():.2f} [{q95(c.rot_deg):.2f}]') if len(c) else ('–', '–')
            out.append(f'| {L}× | {s} | {len(g)} | {g.pivot_mm.median():.2f} [{q95(g.pivot_mm):.2f}] | {cc[0]} | '
                       f'{g.rot_deg.median():.2f} [{q95(g.rot_deg):.2f}] | {cc[1]} | {imp} |')
    return out + ['']


def rater_spread(df):
    out = ['## Spread between raters (stepwise method)', '',
           'Median pivot error per rater (bones pooled): median across the 10 raters [min–max].', '',
           '| Level | ' + ' | '.join(SP) + ' |', '|---|' + '---|' * len(SP)]
    for L in RG['levels']:
        cells = []
        for s in SP:
            d = df[(df.level == str(L)) & (df.arm == RG['arm']) & (df.spacing == s) & df.item.isin(BONES)]
            m = d.groupby('rater').pivot_mm.median()
            cells.append(f'{m.median():.2f} [{m.min():.2f}–{m.max():.2f}]' if len(m) else '–')
        out.append(f'| {L}× | ' + ' | '.join(cells) + ' |')
    return out + ['']


def figure(df, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(len(SP))
    series = [('ideal', 'keyframe_interp', 'k', ':'), ('ideal', 'new_opt_stepwise', 'k', '-')]
    for i, L in enumerate(RG['levels']):
        series += [(str(L), RG['reference_arm'], f'C{i}', ':'), (str(L), RG['arm'], f'C{i}', '-')]
    for lv, arm, col, ls in series:
        d = df[(df.level == lv) & (df.arm == arm) & df.item.isin(BONES)]
        if d.empty:
            continue
        lab = f'{"ideal" if lv == "ideal" else lv + "×"} {"interp" if "interp" in arm else "stepwise"}'
        ax[0].plot(x, [d[d.spacing == s].pivot_mm.median() for s in SP], ls, marker='o', color=col, label=lab)
        ax[1].plot(x, [d[d.spacing == s].rot_deg.median() for s in SP], ls, marker='o', color=col)
        ax[2].plot(x, [100 * needs(d[d.spacing == s])['welte'].mean() for s in SP], ls, marker='o', color=col)
    for a, t in zip(ax, ['Median pivot error (mm)', 'Median rotation error (°)', 'Needs correction, Welte criterion (%)']):
        a.set_xticks(x, SP); a.set_xlabel('Keyframe spacing (frames)'); a.set_title(t)
    ax[0].set_yscale('log'); ax[1].set_yscale('log')
    fig.legend(loc='lower center', ncol=8, frameon=False, fontsize=9)
    fig.suptitle('Simulated raters: interpolation (dotted) and stepwise keyframe-correction method (solid), vs ideal keyframes (black)')
    plt.tight_layout(rect=(0, 0.08, 1, 0.94)); plt.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interp', action='store_true'); ap.add_argument('--assemble', action='store_true')
    a = ap.parse_args()
    for L, r, path in rater_dirs():
        if not (a.interp or a.assemble):
            break
        use_rater(L, r, path)
        args = argparse.Namespace(arm=RG['reference_arm'], spacing=','.join(SP), stage='gaps', task=0, ntasks=1,
                                  max_units=0, quick=False)
        if a.interp:
            kg.cmd_run(args)
        if a.assemble:
            for arm in ARMS:
                args.arm = arm
                try:
                    kg.cmd_assemble(args)
                except FileNotFoundError as e:                # corrected keyframes not there yet
                    kg.log(f'L{L} {r} {arm}: skipped ({e})')
    df, k = score_tracks()
    df = pd.concat([df, ideal_rows()], ignore_index=True)
    os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
    df.to_csv(os.path.join(OUT, 'scores.csv'), index=False); k.to_csv(os.path.join(OUT, 'keyframes.csv'), index=False)
    lines = ['# Keyframe gaps with simulated raters', '',
             f'Levels {RG["levels"]} × {RG["raters"]} raters (rater_model), spacings {SP}. Unfiltered; scored on the '
             'non-keyframes between the bookends, pooled over trials and raters.', '']
    lines += main_table(df) + keyframe_table(k) + bone_table(df) + rater_spread(df)
    s = '\n'.join(lines) + '\n'; open(os.path.join(OUT, 'summary.md'), 'w').write(s); print(s)
    figure(df, os.path.join(OUT, 'figures', 'raters_by_spacing.png'))


if __name__ == '__main__':
    main()
