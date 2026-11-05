"""Time the expensive pieces on the current device, to size the SLURM arrays.

    python methods_comparison/bench.py

Prints the time of one Autoscoper-replica cost evaluation (pso_simplex: about 4,700 per bone-frame in the pilot)
at full and 512 resolution, and of one co-fitted anchored-fit iteration (new_opt: 90 per gap), and an
estimate of the GPU-hours per arm.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import CFG, arm_cfg, f2f_for, keyframes, ref_keys, setup_anchored, spline_track, trial, units  # noqa: E402
from src.costs import autoscoper_cost  # noqa: E402
from src.device import DEVICE  # noqa: E402
from src.drr import View  # noqa: E402
from src.geometry import Pose  # noqa: E402
from src.tracking import AnchoredFit  # noqa: E402


def sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def timeit(fn, n):
    fn(); sync(); t0 = time.time()
    for _ in range(n): fn()
    sync(); return (time.time() - t0) / n


def main():
    T = trial(CFG['trials'][0]); f = 25
    print(f'device {DEVICE}', torch.cuda.get_device_name() if DEVICE.type == 'cuda' else '')
    per_eval = {}
    for render, size in ((T.render_resolution[0], None), (512, 512)):
        views = [View(c, render, T.sample_distance, T.ray_intensity) for c in T.cams]
        rads = [T.image(ci, f, size) for ci in range(len(T.cams))]
        for n in CFG['bones']:
            P = Pose.from_matrix(T.ref[n][f], T.bones[n].pivot)
            x = torch.zeros(6, dtype=torch.float64)
            with torch.no_grad():
                s = timeit(lambda: float(autoscoper_cost(views, rads, T.bones[n], P, x, T.vie)), 10)
            per_eval.setdefault(render, []).append(s)
            print(f'Autoscoper cost, render {render}, {n}: {1000 * s:.1f} ms per evaluation')
    c = arm_cfg('new_opt'); keys = keyframes(T.n_frames, 0)
    fovs, views, sdfs = setup_anchored(T)
    base = spline_track(T, ref_keys(T, keys, CFG['bones'])); a, b = 20, 30; frames = list(range(a, b + 1))
    fit = AnchoredFit(T, views, fovs, f2f_for(T, fovs, base, frames), sdfs, CFG['bones'], [a, b],
                      {(n, ff): base[n][ff] for n in CFG['bones'] for ff in frames}, mu=c['mu'], d_min=c['d_min'],
                      log=lambda *x: None)
    it = {}
    for res in (256, 512):
        def one():
            for p in fit.params(): p.grad = None
            fit.step(res, fit.sample_frames())
        it[res] = timeit(one, 3)
        print(f'Anchored fit (co-fitted, contact on), res {res}: {it[res]:.2f} s per iteration')
    evals = 4700
    nB = sum(1 for u in units('pso_simplex') for _ in range(u['b'] - u['a'] - 1)) * 1.0
    gB = nB * evals * np.mean(per_eval[T.render_resolution[0]]) / 3600
    nC = len(units('new_opt')); gC = nC * (40 * it[256] + 50 * it[512]) / 3600
    print(f'\nEstimate, one process per GPU: pso_simplex {nB:.0f} bone-frames x {evals} evaluations = {gB:.0f} GPU-hours; '
          f'new_opt {nC} gaps = {gC:.1f} GPU-hours. Several processes per GPU (WORKERS) overlap these.')


if __name__ == '__main__':
    main()
