"""Fit the rater-error model to Welte et al. (2022) Figures 4 and 5, and check the fit.

    python rater_model/calibrate.py

Simulates many raters over a 41-frame trial, scores each frame with Welte's metrics (helical rotation
angle; translation along the helical axis) and summarises each rater by the median, quartiles and mean
of their 41 frames, as in Figure 4. Fits, in order:
  1. phi (AR(1) coefficient): lag-1 autocorrelation of the per-frame rotation error, Figure 5.
  2. between (spread of rater quality): coefficient of variation of the three operators' medians, Figure 4.
  3. translation offset share, one value for all bones: Figure 4's translation range relative to the median
     and the spread of the operators' medians, jointly (Figure 5 has no translation, so this replaces assuming
     the rotation share).
  4. scale per bone and quantity: the median of the simulated raters' medians equals the mean of the three
     operators' medians. Errors scale linearly with it, so this is exact.
The means, ranges (Figure 4's vertical lines, which are the frame range: see calibration.md) and Figure 5's
quartiles and lag-10 correlation are not fitted and serve as checks.

Writes results/rater_model/params.json, calibration.md and figures/calibration.png.
"""
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
sys.path.insert(0, ROOT)
from src.raters import draw_track, helical_metrics  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, 'rater_model', 'config.yaml')))
OUT = os.path.join(ROOT, 'results', 'rater_model'); os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
BONES = CFG['bones']; NF = CFG['assessment_frames']
Q = {'rotation_deg': 'Rotation (°)', 'translation_mm': 'Translation along helical axis (mm)'}


def welte():
    f4 = pd.read_csv(os.path.join(ROOT, CFG['welte_fig4']))
    f4['bone'] = f4.bone.map({v: k for k, v in BONES.items()})
    f5 = pd.read_csv(os.path.join(ROOT, CFG['welte_fig5']))
    return f4, f5


def simulate(scale_rot, scale_trans, share, phi, between, n, seed, share_trans=None):
    """n simulated raters over NF frames: dict of arrays (n, NF) per metric."""
    rng = np.random.default_rng(seed)
    rot, ax, piv = [], [], []
    st = share if share_trans is None else share_trans
    for _ in range(n):
        r, t = draw_track(rng, NF, scale_rot, scale_trans, share, st, phi, between)
        a, h, p = helical_metrics(r, t); rot.append(a); ax.append(h); piv.append(p)
    return dict(rotation_deg=np.array(rot), translation_mm=np.array(ax), pivot_mm=np.array(piv))


def lag(x, k):
    return np.corrcoef(x[:-k], x[k:])[0, 1]


def summarise(x):
    """Per-rater summaries over frames: median, mean, range, quartiles."""
    return dict(median=np.median(x, 1), mean=x.mean(1), min=x.min(1), max=x.max(1),
                q25=np.percentile(x, 25, 1), q75=np.percentile(x, 75, 1))


def main():
    f4, f5 = welte()
    n, seed = CFG['n_sim_raters'], CFG['seed']
    fig5 = {'cal': f5.calcaneus_rotation_bias_deg.values, 'tal': f5.talus_rotation_bias_deg.values,
            'tib': f5.tibia_rotation_bias_deg.values[:33]}            # tibia before the occlusion episode
    lag1_obs = np.mean([lag(v, 1) for v in fig5.values()])

    # 1. phi: lag-1 autocorrelation of the rotation magnitude series (shape only, so scale = 1, between = 0)
    phi_fit = {}
    for phi in CFG['phi_grid']:
        l1 = []
        for b in BONES:
            sim = simulate(1.0, 1.0, CFG['offset_share'][b], phi, 0.0, 400, seed)['rotation_deg']
            l1.append(np.nanmedian([lag(x, 1) for x in sim]))
        phi_fit[phi] = np.mean(l1)
    phi = min(phi_fit, key=lambda p: abs(phi_fit[p] - lag1_obs))

    # 2. between: CV of rater medians, pooled over bones, per quantity
    cv_obs = {q: np.mean([f4[(f4.bone == b) & (f4.quantity == q)]['median'].std(ddof=1) /
                          f4[(f4.bone == b) & (f4.quantity == q)]['median'].mean() for b in BONES]) for q in Q}
    between, cv_fit = {}, {}
    for q in Q:
        res = {}
        for bt in CFG['between_grid']:
            cvs = []
            for b in BONES:
                m = np.median(simulate(1.0, 1.0, CFG['offset_share'][b], phi, bt, 1000, seed)[q], 1)
                # CV of 3 operators: expected sample CV of triplets drawn from the simulated raters
                trip = m[: (len(m) // 3) * 3].reshape(-1, 3)
                cvs.append(np.mean(trip.std(1, ddof=1) / trip.mean(1)))
            res[bt] = np.mean(cvs)
        between[q] = min(res, key=lambda v: abs(res[v] - cv_obs[q])); cv_fit[q] = res

    # 3. translation offset share (no per-frame translation data). One value for all bones, fitted jointly to
    #    Figure 4's translation range relative to the median (max / median, the within-rater spread) and to the
    #    spread of the three operators' medians (CV, the between-rater spread): minimise the summed squared log
    #    ratios over bones. Per-bone values are not identifiable from three operators.
    obj, detail = {}, {}
    for sh in CFG['share_trans_grid']:
        tot, det = 0.0, {}
        for b in BONES:
            w = f4[(f4.bone == b) & (f4.quantity == 'translation_mm')]
            r_obs = np.mean(w.vertical_line_high / w['median']); cv_obs_b = w['median'].std(ddof=1) / w['median'].mean()
            x = simulate(1.0, 1.0, CFG['offset_share'][b], phi, between['translation_mm'], 1500, seed, sh)['translation_mm']
            m = np.median(x, 1); trip = m.reshape(-1, 3)
            r_sim = np.median(x.max(1) / m); cv_sim = np.mean(trip.std(1, ddof=1) / trip.mean(1))
            tot += np.log(r_sim / r_obs) ** 2 + np.log(cv_sim / cv_obs_b) ** 2
            det[b] = (r_obs, r_sim, cv_obs_b, cv_sim)
        obj[sh], detail[sh] = tot, det
    sh_t = min(obj, key=obj.get)
    share_t = {b: sh_t for b in BONES}; ratio_fit = detail[sh_t]

    # 4. scales: exact, since every metric scales linearly with the scale
    params = dict(phi=phi, between=between, offset_share=CFG['offset_share'], offset_share_trans=share_t, scale={})
    sims = {}
    for b in BONES:
        s1 = {q: simulate(1.0, 1.0, CFG['offset_share'][b], phi, between[q], n, seed + i, share_t[b])
              for i, q in enumerate(Q)}
        sc = {}
        for q in Q:
            target = f4[(f4.bone == b) & (f4.quantity == q)]['median'].mean()
            sc[q] = target / np.median(np.median(s1[q][q], 1))
        params['scale'][b] = dict(rotation_deg=float(sc['rotation_deg']), translation_mm=float(sc['translation_mm']))
        sims[b] = simulate(sc['rotation_deg'], sc['translation_mm'], CFG['offset_share'][b], phi, between['rotation_deg'],
                           n, seed + 10, share_t[b])
        # translation spread uses its own 'between': rerun for the translation metrics
        st = simulate(sc['rotation_deg'], sc['translation_mm'], CFG['offset_share'][b], phi, between['translation_mm'],
                      n, seed + 11, share_t[b])
        sims[b]['translation_mm'] = st['translation_mm']; sims[b]['pivot_mm'] = st['pivot_mm']
    json.dump(params, open(os.path.join(OUT, 'params.json'), 'w'), indent=1)

    # ---------------- checks
    lines = ['# Rater model calibration', '',
             'Model and procedure: `src/raters.py`, `rater_model/calibrate.py`. Welte et al. (2022) values digitised '
             'from Figures 4 and 5. Simulated values: median across 4,000 simulated raters of each per-rater summary, '
             'with the 5th–95th percentile across raters in brackets.', '',
             'Note on the digitised Figure 4: the "vertical line" values recorded as low/high match the minimum and '
             'maximum of New Operator 2\'s frames in Figure 5 (calcaneus 0.29/1.26 vs 0.29/1.28; tibia low 0.48 vs 0.48), '
             'not the interquartile range. They are compared here with each simulated rater\'s minimum and maximum over '
             'the 41 frames.', '',
             f'**Fitted:** AR(1) coefficient φ = {phi} (simulated lag-1 autocorrelation {phi_fit[phi]:.2f}, '
             f'Figure 5 {lag1_obs:.2f}); spread of rater quality (log-normal SD) {between["rotation_deg"]} for rotation '
             f'(operators\' CV {cv_obs["rotation_deg"]:.2f}, simulated {cv_fit["rotation_deg"][between["rotation_deg"]]:.2f}) '
             f'and {between["translation_mm"]} for translation (operators\' CV {cv_obs["translation_mm"]:.2f}, simulated '
             f'{cv_fit["translation_mm"][between["translation_mm"]]:.2f}). Translation offset share {sh_t}, one value '
             'for all bones, fitted jointly to Figure 4\'s translation range and the spread of the operators\' medians: ' +
             ', '.join(f'{BONES[b]} max/median {ratio_fit[b][0]:.1f} Welte vs {ratio_fit[b][1]:.1f} simulated, CV of medians '
                       f'{ratio_fit[b][2]:.2f} vs {ratio_fit[b][3]:.2f}' for b in BONES) + '. Within-rater range and between-rater '
             'spread pull in opposite directions (a larger offset share narrows the first and widens the second), and with '
             'three operators per bone neither is precise, so this is a compromise.', '',
             '| Bone | Quantity | Scale (σ per axis) | Statistic | Welte: three operators | Welte mean | Simulated |',
             '|---|---|---|---|---|---|---|']
    for b, name in BONES.items():
        for q in Q:
            w = f4[(f4.bone == b) & (f4.quantity == q)]
            s = summarise(sims[b][q])
            for stat, col in (('median', 'median'), ('mean', 'mean'), ('min', 'vertical_line_low'), ('max', 'vertical_line_high')):
                vals = w[col].dropna().values
                sv = s[stat]
                lines.append(f'| {name} | {q.split("_")[0]} | {params["scale"][b][q]:.3f} | {stat} | '
                             f'{", ".join(f"{v:.2f}" for v in vals)} | {vals.mean():.2f} | '
                             f'{np.median(sv):.2f} [{np.percentile(sv, 5):.2f}–{np.percentile(sv, 95):.2f}] |')
    lines += ['', '## Implied pivot displacement (not observable in Welte)', '',
              'Pivot displacement |t| per frame under the isotropic assumption, trained level (1×).', '',
              '| Bone | Median mm | 75th pct mm | 95th pct mm | Helical translation median mm (Welte metric) |', '|---|---|---|---|---|']
    for b, name in BONES.items():
        p = sims[b]['pivot_mm'].ravel(); h = sims[b]['translation_mm'].ravel()
        lines.append(f'| {name} | {np.median(p):.2f} | {np.percentile(p, 75):.2f} | {np.percentile(p, 95):.2f} | {np.median(h):.2f} |')
    lines += ['', '## Figure 5 checks: within-rater spread and time structure (rotation)', '',
              'Figure 5 gives New Operator 2\'s rotation error at every frame, so the true quartiles of one operator are known. '
              'Simulated: raters with a median close to that operator\'s (within ±15 %).', '',
              '| Bone | Welte NO2 q25 / median / q75 ° | Simulated q25 / median / q75 ° | Welte lag-1 / lag-10 | Simulated lag-1 / lag-10 |',
              '|---|---|---|---|---|']
    for b, name in BONES.items():
        x = fig5[b]; sim = sims[b]['rotation_deg']; med = np.median(sim, 1)
        near = sim[np.abs(med / np.median(x) - 1) < 0.15]
        sq = [np.median(np.percentile(near, p, 1)) for p in (25, 50, 75)]
        lines.append(f'| {name} | {np.percentile(x, 25):.2f} / {np.median(x):.2f} / {np.percentile(x, 75):.2f} | '
                     f'{sq[0]:.2f} / {sq[1]:.2f} / {sq[2]:.2f} | {lag(x, 1):.2f} / {lag(x, 10):.2f} | '
                     f'{np.nanmedian([lag(v, 1) for v in near]):.2f} / {np.nanmedian([lag(v, 10) for v in near]):.2f} |')
    lines += ['', 'Rotation share > 2° and translation share > 1 mm across all simulated frames (trained level): ' +
              ', '.join(f'{name} {(sims[b]["rotation_deg"] > 2).mean():.3f} / {(sims[b]["translation_mm"] > 1).mean():.3f}'
                        for b, name in BONES.items()) + '.']
    open(os.path.join(OUT, 'calibration.md'), 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
    figure(f4, f5, sims, params)


def figure(f4, f5, sims, params):
    """Welte operators (median, IQR) next to 12 simulated trained raters, per bone and quantity; plus Figure 5."""
    fig, ax = plt.subplots(3, 3, figsize=(15, 11), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    ops = ['New operator 1', 'New operator 2', 'Expert']
    for i, (b, name) in enumerate(BONES.items()):
        for j, q in enumerate(Q):
            a = ax[i, j]; w = f4[(f4.bone == b) & (f4.quantity == q)].set_index('rater')
            for k, o in enumerate(ops):                    # Welte: median and range of the 41 frames
                a.errorbar(k, w.loc[o, 'median'], yerr=[[w.loc[o, 'median'] - w.loc[o, 'vertical_line_low']],
                                                         [w.loc[o, 'vertical_line_high'] - w.loc[o, 'median']]],
                           fmt='o', color='C3', capsize=3, lw=2)
            sim = sims[b][q][:12]
            for k, x in enumerate(sim):
                a.violinplot(x, positions=[4 + k], widths=0.8, showextrema=False)
                a.errorbar(4 + k, np.median(x), yerr=[[np.median(x) - x.min()], [x.max() - np.median(x)]],
                           fmt='o', color='C0', capsize=2, lw=1.5, ms=4)
            a.axvline(3, color='0.8', lw=0.8); a.set_xticks([1, 9.5], ['Welte operators', 'Simulated trained raters'])
            a.set_title(f'{name}: {Q[q]}', fontsize=10)
        a = ax[i, 2]; col = {'cal': 'calcaneus', 'tal': 'talus', 'tib': 'tibia'}[b] + '_rotation_bias_deg'
        a.plot(f5.frame_index, f5[col], 'o-', color='C3', label='Welte New operator 2')
        rng = np.random.default_rng(5)
        from src.raters import draw_track
        for k in range(3):
            r, t = draw_track(rng, len(f5), params['scale'][b]['rotation_deg'], params['scale'][b]['translation_mm'],
                              params['offset_share'][b], params['offset_share_trans'][b], params['phi'], params['between']['rotation_deg'])
            a.plot(f5.frame_index, np.linalg.norm(r, axis=1), '-', color='C0', alpha=.6, label='Simulated raters' if k == 0 else None)
        a.set_title(f'{name}: rotation error by frame (Figure 5)', fontsize=10); a.set_xlabel('Frame')
        if i == 0: a.legend(fontsize=8)
    fig.suptitle('Rater model calibration: Welte et al. (2022) operators (red) vs simulated trained raters (blue); points = median, bars = range of 41 frames')
    plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(os.path.join(OUT, 'figures', 'calibration.png'), dpi=110)


if __name__ == '__main__':
    main()
