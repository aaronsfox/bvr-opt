"""Keyframe-correction step test: original (steps scale with the tolerance) vs fixed-size steps.

    python rater_gaps/tests/kc_step_test.py run --step tol|mm --level 4 --rater r00 --spacing full
    python rater_gaps/tests/kc_step_test.py summary

`run` corrects every keyframe of one simulated rater at one spacing (all four trials) with the level's
tolerance, the chosen step parameterisation and otherwise the new_opt_stepwise_kc settings, and writes one
row per keyframe and bone (given and corrected error against the reference) to
results/rater_gaps/tests/kc_step/<step>_L<level>_<rater>_s<spacing>.csv. `summary` compares the two.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'keyframe_gaps')); sys.path.insert(1, ROOT)
import run as kg  # noqa: E402
from src.geometry import helical_error, pivot_error  # noqa: E402
from src.pipeline import anchored_setup, f2f_for, spline_track  # noqa: E402
from src.tracking import KeyframeFit  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'rater_gaps', 'tests', 'kc_step')
RG = yaml.safe_load(open(os.path.join(ROOT, 'rater_gaps', 'config.yaml')))


def run(a):
    kg.SOURCE['keyframes'] = f'rater:L{a.level}/{a.rater}'
    c = kg.CFG['arms']['new_opt_stepwise_kc']; kc = c['keyframe_correction']; tol = RG['kc_tol'][a.level]
    rows = []; os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{a.step}_L{a.level}_{a.rater}_s{a.spacing}.csv')
    for tn in kg.CFG['trials']:
        T = kg.trial(tn); keys = kg.keyframes(T.n_frames, a.spacing); B = kg.BONES
        given = kg.given_keyframes(T, keys); fovs, views, sdfs = anchored_setup(T, B); sp = spline_track(T, given)
        for k in keys:
            frames = list(range(max(0, k - kc['window']), min(T.n_frames - 1, k + kc['window']) + 1))
            base = {(n, f): sp[n][f] if np.isfinite(sp[n][f]).all() else given[n][k] for n in B for f in frames}
            f2f = f2f_for(T, fovs, {n: {f: base[(n, f)] for f in frames} for n in B}, frames, B, margin=150)
            t0 = time.time()
            out, _ = KeyframeFit(T, views, fovs, f2f, sdfs, B, k, frames, base, tol=tol, w_drr=c['w_drr'],
                                 w_f2f=c['w_f2f'], mu=c['mu'], d_min=c['d_min'], key_w=kc['key_w'], res=kc['res'],
                                 iters=kc['iters'], lrs=kc['lrs'], step=a.step, log=lambda *x: None).run()
            for n in B:
                G = T.ref[n][k]; piv = T.bones[n].pivot
                rows.append(dict(step=a.step, level=a.level, rater=a.rater, spacing=a.spacing, trial=tn, frame=k, bone=n,
                                 given_mm=pivot_error(given[n][k], G, piv), given_deg=helical_error(given[n][k], G)[0],
                                 corr_mm=pivot_error(out[n], G, piv), corr_deg=helical_error(out[n], G)[0]))
            pd.DataFrame(rows).to_csv(path, index=False)            # written after every keyframe
            print(f'{a.step} L{a.level} {a.rater} {tn} keyframe {k}: {time.time() - t0:.0f}s', flush=True)


def summary():
    d = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(OUT, '*.csv'))])
    q = lambda x: np.percentile(x, 95)
    lines = ['| Level | Step | Keyframes | Given pivot mm | Corrected pivot mm | Given rotation ° | Corrected rotation ° | Pivot improved % | Rotation improved % |',
             '|---|---|---|---|---|---|---|---|---|']
    for (L, st), g in d.groupby(['level', 'step']):
        lines.append(f'| {L}× | {st} | {len(g)} | {g.given_mm.median():.2f} [{q(g.given_mm):.2f}] | {g.corr_mm.median():.2f} [{q(g.corr_mm):.2f}] | '
                     f'{g.given_deg.median():.2f} [{q(g.given_deg):.2f}] | {g.corr_deg.median():.2f} [{q(g.corr_deg):.2f}] | '
                     f'{100 * (g.corr_mm < g.given_mm).mean():.0f} | {100 * (g.corr_deg < g.given_deg).mean():.0f} |')
    s = 'Median [95th percentile] over keyframes and bones.\n\n' + '\n'.join(lines) + '\n'
    open(os.path.join(OUT, 'summary.md'), 'w').write(s); print(s)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'summary']); ap.add_argument('--step', default='tol')
    ap.add_argument('--level', type=int, default=4); ap.add_argument('--rater', default='r00'); ap.add_argument('--spacing', default='full')
    a = ap.parse_args()
    run(a) if a.cmd == 'run' else summary()
