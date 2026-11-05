"""Check of the keyframe tolerance on hop108_0001, keyframes 10-50 (four gaps fitted together).

Keyframes are the reference poses with a rater-like error of exactly 1 mm (pivot) and 2 deg (random axis)
per bone and keyframe, i.e. at Welte's operator standard; all trial keyframes (0..70, 73) carry error so the
spline path is built from erroneous keyframes too. Runs (arm C settings otherwise):
    hard      key_tol = 0 (keyframes held where the rater put them)
    tol1      key_tol = (1 mm, 2 deg)  (matched to the error)
    tol2      key_tol = (2 mm, 4 deg)  (opened wider)
    perfect   reference keyframes, key_tol = (1 mm, 2 deg): should stay at the truth
Scores keyframes and hidden frames against the reference. Writes results/keytol_<run>.npz/.json.

Usage: python keytol_check.py <run>
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'methods_comparison'))
import torch  # noqa: E402

from run import arm_cfg, f2f_for, setup_anchored, spline_track, trial  # noqa: E402
from bvr_opt.geometry import helical_error, pivot_error, rotvec_to_matrix  # noqa: E402
from bvr_opt.tracking import AnchoredFit  # noqa: E402

RUNS = {'hard': (True, None), 'tol1': (True, (1.0, 2.0)), 'tol2': (True, (2.0, 4.0)), 'perfect': (False, (1.0, 2.0))}
BONES = ['tib', 'tal', 'cal']; WIN = [10, 20, 30, 40, 50]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def perturb(M, pivot, rng, mm=1.0, deg=2.0):
    """Rigid error about the bone pivot: pivot moved `mm` in a random direction, rotated `deg` about a random axis."""
    u = rng.normal(size=3); u /= np.linalg.norm(u); a = rng.normal(size=3); a /= np.linalg.norm(a)
    dR = rotvec_to_matrix(torch.tensor(a * deg)).cpu().numpy()
    R = M[:3, :3]; p = R @ pivot + M[:3, 3]
    R2 = R @ dR; p2 = p + mm * u
    out = np.eye(4); out[:3, :3] = R2; out[:3, 3] = p2 - R2 @ pivot
    return out


def main(run):
    torch.set_num_threads(int(os.environ.get('THREADS', '2')))
    noisy, tol = RUNS[run]
    T = trial('T0014_hop108_0001'); N = T.n_frames
    keys_all = list(range(0, N, 10)) + [N - 1]
    rng = np.random.default_rng(7)
    keyp = {n: {k: (perturb(T.ref[n][k], T.bones[n].pivot, rng) if noisy else T.ref[n][k]) for k in keys_all} for n in BONES}
    base = spline_track(T, keyp)
    c = arm_cfg('C'); fovs, views, sdfs = setup_anchored(T)
    frames = list(range(WIN[0], WIN[-1] + 1))
    fit = AnchoredFit(T, views, fovs, f2f_for(T, fovs, base, frames), sdfs, BONES, WIN,
                      {(n, f): base[n][f] for n in BONES for f in frames}, mu=c['mu'], d_min=c['d_min'],
                      key_tol=tol, seed=0, log=log)
    track, info = fit.run()
    tr = {n: base[n].copy() for n in BONES}
    for (n, f), M in track.items(): tr[n][f] = M
    rows = {}
    for n in BONES:
        for kind, fs in (('key', WIN), ('hidden', [f for f in frames if f not in WIN])):
            for src, X in (('input', base[n]), ('fit', tr[n])):
                e = [pivot_error(X[f], T.ref[n][f], T.bones[n].pivot) for f in fs]
                r = [helical_error(X[f], T.ref[n][f])[0] for f in fs]
                rows[f'{n}_{kind}_{src}'] = dict(pivot_med=float(np.median(e)), pivot_max=float(np.max(e)),
                                                 rot_med=float(np.median(r)), rot_max=float(np.max(r)))
    np.savez(os.path.join(HERE, 'results', f'keytol_{run}.npz'), **tr)
    info.pop('hist', None)
    json.dump(dict(scores=rows, info=info), open(os.path.join(HERE, 'results', f'keytol_{run}.json'), 'w'), indent=1)
    for k, v in rows.items():
        log(run, k, ' '.join(f'{a} {b:.2f}' for a, b in v.items()))


if __name__ == '__main__':
    main(sys.argv[1])
