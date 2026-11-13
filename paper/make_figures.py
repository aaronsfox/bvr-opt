"""Figures and tables for the short communication, from the committed rater-gaps results.

    python paper/make_figures.py

Measures follow the paper's analysis: Welte et al.'s (2022) helical rotation and translation along the
helical axis, and their operator criterion for frames needing correction (rotation > 2 deg or translation > 1 mm).
Scored frames: every non-keyframe between the first and last keyframes; bones pooled.
Medians carry a 95% CI bootstrapped over raters (2,000 resamples); reference keyframes have no CI.

Writes results/paper/: figure2_spacing.{png,pdf}, figure3_refitting.{png,pdf}, table1.{csv,md,xlsx},
table_refitting.{csv,md}, checks.md.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import plotstyle as ps  # noqa: E402

RG = os.path.join(ROOT, 'results', 'rater_gaps')
OUT = os.path.join(ROOT, 'results', 'paper'); os.makedirs(OUT, exist_ok=True)
BONES = ['tib', 'tal', 'cal']
SP = ['5', '10', '20', '30', '40', 'full']
SPL = ['5', '10', '20', '30', '40', 'Trial']
GROUPS = [('ideal', 'Reference'), ('1', '1x'), ('2', '2x'), ('4', '4x')]
METHOD = {'ideal': 'new_opt_stepwise', '1': 'new_opt_stepwise_kc', '2': 'new_opt_stepwise_kc', '4': 'new_opt_stepwise_kc'}
INTERP = 'keyframe_interp'
TH_ROT, TH_TR = 2.0, 1.0
N_BOOT = 2000


def load():
    d = pd.read_csv(os.path.join(RG, 'scores.csv'), dtype={'spacing': str, 'level': str})
    d = d[d.item.isin(BONES)].copy()
    d['bad'] = (d.rot_deg > TH_ROT) | (d.hel_mm > TH_TR)
    k = pd.read_csv(os.path.join(RG, 'keyframes.csv'), dtype={'spacing': str, 'level': str})
    k = k[k.item.isin(BONES)].copy()
    k['bad'] = (k.rot_deg > TH_ROT) | (k.hel_mm > TH_TR)
    return d, k


def boot_ci(e, col, rng):
    groups = [g[col].values for _, g in e.groupby('rater')]
    if len(groups) < 2:
        return np.nan, np.nan
    b = [np.median(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(N_BOOT)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def summarise(d):
    rng = np.random.default_rng(2022); rows = []
    for lv, lab in GROUPS:
        for meth, arm in (('Refinement', METHOD[lv]), ('Interpolation', INTERP)):
            for s, sl in zip(SP, SPL):
                e = d[(d.level == lv) & (d.arm == arm) & (d.spacing == s)]
                r = dict(group=lab, method=meth, spacing=sl, n=len(e))
                for col, q in (('rot_deg', 'rot'), ('hel_mm', 'tr')):
                    lo, hi = boot_ci(e, col, rng)
                    r.update({f'{q}_med': e[col].median(), f'{q}_lo': lo, f'{q}_hi': hi, f'{q}_p95': e[col].quantile(.95)})
                r['needs_pct'] = 100 * e.bad.mean()
                rows.append(r)
    return pd.DataFrame(rows)


def cell(r, q, nd=2):
    ci = '' if np.isnan(r[f'{q}_lo']) else f' ({r[f"{q}_lo"]:.{nd}f}–{r[f"{q}_hi"]:.{nd}f})'
    return f'{r[f"{q}_med"]:.{nd}f}{ci} [{r[f"{q}_p95"]:.{nd}f}]'


def table1(t):
    rows = []
    for (g, s), x in t.groupby(['group', 'spacing'], sort=False):
        a = x[x.method == 'Refinement'].iloc[0]; b = x[x.method == 'Interpolation'].iloc[0]
        rows.append({'Keyframes': g, 'Spacing (frames)': s,
                     'Rotation (°): refinement': cell(a, 'rot'), 'Rotation (°): interpolation': cell(b, 'rot'),
                     'Translation (mm): refinement': cell(a, 'tr'), 'Translation (mm): interpolation': cell(b, 'tr'),
                     'Needs correction (%): refinement': f'{a.needs_pct:.1f}', 'Needs correction (%): interpolation': f'{b.needs_pct:.1f}'})
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'table1.csv'), index=False)
    try:
        T.to_excel(os.path.join(OUT, 'table1.xlsx'), index=False)
    except Exception as ex:  # openpyxl missing
        print('xlsx skipped:', ex)
    note = ('Table 1. Tracking error between keyframes by keyframe source and spacing: median (95% CI) [95th percentile] '
            'of helical rotation and of translation along the helical axis, and the percentage of bone-frames needing '
            'correction (rotation > 2° or translation > 1 mm; Welte et al., 2022). Bones, frames, trials and raters pooled; '
            'CIs bootstrapped over the 10 raters (none for reference keyframes). "Trial" = keyframes at the first and last '
            'tracked frames only. Refinement with reference keyframes omits keyframe refitting.')
    with open(os.path.join(OUT, 'table1.md'), 'w') as fh:
        fh.write(note + '\n\n' + T.to_markdown(index=False) + '\n')
    return T


def figure2(t):
    """Needs correction, median rotation and median translation against spacing; colour/marker = keyframe source,
    solid filled = refinement, dotted open = interpolation."""
    ps.apply()
    fig, ax = plt.subplots(1, 3, figsize=(ps.DOUBLE, 2.7))
    x = np.arange(len(SP))
    for i, (_, lab) in enumerate(GROUPS):
        for meth in ('Interpolation', 'Refinement'):
            e = t[(t.group == lab) & (t.method == meth)].set_index('spacing').loc[SPL]
            interp = meth == 'Interpolation'
            kw = dict(ls=':' if interp else '-', lw=1.2 if interp else 1.6, **ps.style(i, hollow=interp))
            ax[0].plot(x, e.needs_pct, label=f'{lab}, {meth.lower()}', **kw)
            for a, q in ((ax[1], 'rot'), (ax[2], 'tr')):
                a.plot(x, e[f'{q}_med'], **kw)
                if not interp and not e[f'{q}_lo'].isna().all():
                    a.fill_between(x, e[f'{q}_lo'], e[f'{q}_hi'], color=ps.color(i), alpha=0.2, lw=0)
    ax[0].set_ylim(0, 100); ax[0].set_ylabel('Frames needing correction (%)')
    ax[1].set_ylabel('Helical rotation error (°)'); ax[2].set_ylabel('Helical translation error (mm)')
    for a in ax[1:]:
        a.set_yscale('log'); ps.plain_log(a)
    ax[1].axhline(TH_ROT, color='0.6', ls='--', lw=0.8); ax[2].axhline(TH_TR, color='0.6', ls='--', lw=0.8)
    for a, l in zip(ax, 'abc'):
        a.set_xticks(x, SPL); a.set_xlabel('Keyframe spacing (frames)')
        a.text(-0.2, 1.04, l, transform=a.transAxes, fontsize=10, fontweight='bold')
    h, lab = ax[0].get_legend_handles_labels()
    order = [k for k in range(len(h)) if 'refinement' in lab[k]] + [k for k in range(len(h)) if 'interpolation' in lab[k]]
    fig.legend([h[k] for k in order], [lab[k] for k in order], loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1)); ps.save(fig, os.path.join(OUT, 'figure2_spacing.png')); plt.close(fig)


KF_SPACING = '5'   # each keyframe position once (given keyframes repeat across spacings)


def refitting(k):
    k = k[k.spacing == KF_SPACING]
    rows = []
    for lv, lab in GROUPS[1:]:
        r = dict(group=lab)
        for kind, kl in (('given', 'Given'), ('corrected', 'Refitted')):
            e = k[(k.level == lv) & (k.kind == kind)]
            r[f'{kl} rotation (°)'] = f'{e.rot_deg.median():.2f} [{e.rot_deg.quantile(.95):.2f}]'
            r[f'{kl} translation (mm)'] = f'{e.hel_mm.median():.2f} [{e.hel_mm.quantile(.95):.2f}]'
            r[f'{kl}: needs correction (%)'] = f'{100 * e.bad.mean():.1f}'
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, 'table_refitting.csv'), index=False)
    with open(os.path.join(OUT, 'table_refitting.md'), 'w') as fh:
        fh.write('Keyframe error before (given) and after refitting: median [95th percentile]; 5-frame spacing (every '
                 'keyframe position once; 1,200 bone-keyframes per group), bones and raters pooled. Other spacings give '
                 'medians within 0.05 deg and 0.03 mm.\n\n' + T.to_markdown(index=False) + '\n')
    return T


def figure3(k):
    """Keyframe error given vs refitted, by rater group (5-frame spacing, each keyframe once): every bone-keyframe
    as a jittered point, with a box (IQR, median; whiskers 5th-95th percentile) on top. Threshold lines dashed."""
    ps.apply()
    k = k[k.spacing == KF_SPACING]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(ps.DOUBLE * 0.8, 2.8))
    for a, col, th, yl in ((ax[0], 'rot_deg', TH_ROT, 'Helical rotation error (°)'), (ax[1], 'hel_mm', TH_TR, 'Helical translation error (mm)')):
        for i, (lv, lab) in enumerate(GROUPS[1:], start=1):
            for j, kind in enumerate(('given', 'corrected')):
                v = k[(k.level == lv) & (k.kind == kind)][col].values
                pos = 3 * (i - 1) + j
                xj = pos + rng.uniform(-0.3, 0.3, len(v))
                a.scatter(xj, v, s=1.5, marker=ps.marker(i), linewidths=0,
                          color=ps.color(i), alpha=0.25 if kind == 'given' else 0.45, zorder=1, rasterized=True)
                bp = a.boxplot([v], positions=[pos], widths=0.7, whis=(5, 95), showfliers=False, patch_artist=True,
                               medianprops=dict(color='black', lw=1.3), whiskerprops=dict(color='black', lw=0.9),
                               capprops=dict(color='black', lw=0.9), zorder=2)
                bp['boxes'][0].set(facecolor='none', edgecolor='black', lw=1.0, ls='--' if kind == 'given' else '-')
        a.axhline(th, color='0.5', ls=':', lw=0.9, zorder=0)
        a.set_xticks([0, 1, 3, 4, 6, 7], ['Given', 'Refit'] * 3, fontsize=7)
        for i, (_, lab) in enumerate(GROUPS[1:]):
            a.text(3 * i + 0.5, -0.11, lab, transform=a.get_xaxis_transform(), ha='center', va='top', fontsize=9, fontweight='bold')
        a.set_ylabel(yl); a.set_xlim(-0.8, 7.8); a.set_ylim(0, None)
    for a, l in zip(ax, 'ab'):
        a.text(-0.2, 1.04, l, transform=a.transAxes, fontsize=10, fontweight='bold')
    fig.tight_layout(); fig.subplots_adjust(bottom=0.17); ps.save(fig, os.path.join(OUT, 'figure3_refitting.png')); plt.close(fig)


def checks(d, k):
    """Share of 4x refinement frames needing correction that lie in tracks with an unrecovered keyframe (Welte)."""
    kb = k[k.kind == 'corrected'].groupby(['level', 'rater', 'spacing', 'trial', 'item']).bad.any().rename('kbad')
    m = d[d.arm == METHOD['4']].join(kb, on=['level', 'rater', 'spacing', 'trial', 'item'])
    lines = ['# Checks for the results text', '',
             'Share of refinement frames needing correction (Welte criterion) that lie in a track (rater, trial, bone, '
             'spacing) with at least one refitted keyframe still beyond the criterion.', '',
             '| Rater group | ' + ' | '.join(SPL) + ' |', '|---|' + '---|' * len(SP)]
    for lv in ('1', '2', '4'):
        vals = []
        for s in SP:
            e = m[(m.level == lv) & (m.spacing == s) & m.bad]
            vals.append(f'{100 * e.kbad.mean():.0f}% (n={len(e)})' if len(e) else '–')
        lines.append(f'| {lv}x | ' + ' | '.join(vals) + ' |')
    lines += ['', 'Frames needing correction by bone, refinement, pooled over spacings (%):', '']
    b = (m.groupby(['level', 'item']).bad.mean() * 100).round(1).unstack()[BONES]
    lines.append(b.to_markdown())
    s = '\n'.join(lines) + '\n'; open(os.path.join(OUT, 'checks.md'), 'w').write(s); print(s)


def main():
    d, k = load()
    t = summarise(d); t.to_csv(os.path.join(OUT, 'summary_long.csv'), index=False)
    table1(t); figure2(t); refitting(k); figure3(k); checks(d, k)
    print('written to', OUT)


if __name__ == '__main__':
    main()
