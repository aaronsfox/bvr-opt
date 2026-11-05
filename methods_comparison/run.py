"""Methods comparison runner (Analysis 1 in the plan doc): splits every arm into independent units of work and runs them.

    python methods_comparison/run.py list   [--arm keyframe_interp,pso_simplex,new_opt]            # print the units (one per line)
    python methods_comparison/run.py run    --arm pso_simplex [--task 0 --ntasks 1] [--max-units N] [--quick]
    python methods_comparison/run.py assemble                        # units -> one track per (arm, trial, offset)

A unit is the smallest independent piece of an arm:
    keyframe_interp(_linear)              (trial, offset)               interpolation through the keyframes
    pso_simplex, pso_simplex_512          (trial, offset, bone, gap)    forward pass from one keyframe to the next
    pso_simplex_single_pass               (trial, offset, bone)         one forward pass from the first keyframe
    new_opt, new_opt_drr_only, ...        (trial, offset, gap)          anchored fit of one gap, bones co-fitted
    new_opt_per_frame                     (trial, offset, gap)          each frame of the gap fitted on its own

`run` processes units[task::ntasks] and skips units whose output exists, so jobs can be resubmitted.
Outputs go to $BVR_RESULTS (default <repo>/results/methods_comparison), one folder per arm:
<arm>/units/<unit>.json and, after `assemble`, <arm>/tracks/<trial>_o<offset>.npz with {bone: (n_frames, 4, 4)}: keyframes hold the given
(reference) poses unless the arm moved or re-tracked them; NaN = not produced.
"""
import argparse
import json
import os
import socket
import sys
import time
import zlib

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                 # source code is the package src/
import torch  # noqa: E402

from src import collision  # noqa: E402
from src.costs import F2F, autoscoper_cost, fov_masks  # noqa: E402
from src.device import DEVICE  # noqa: E402
from src.drr import View  # noqa: E402
from src.geometry import Pose, interpolate  # noqa: E402
from src.io import Trial  # noqa: E402
from src.optim import autoscoper_optimize  # noqa: E402
from src.tracking import AnchoredFit, per_frame_fit  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, 'methods_comparison', 'config.yaml')))
QUICK = {}   # smoke-test overrides set from the command line (never used for real runs)
RESULTS = os.environ.get('BVR_RESULTS', os.path.join(ROOT, 'results', 'methods_comparison'))


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def arm_cfg(arm):
    c = dict(CFG['arms'][arm])
    if 'base' in c:
        c = {**CFG['arms'][c.pop('base')], **c}
    return c


def keyframes(n_frames, offset):
    return list(range(offset, n_frames, CFG['spacing']))


_trials = {}


def trial(name):
    if name not in _trials:
        _trials[name] = Trial(os.path.join(ROOT, CFG['data_root'], name))
    return _trials[name]


# ------------------------------------------------------------------------------------------- units
def units(arm):
    c = arm_cfg(arm); out = []
    for tn in CFG['trials']:
        N = trial(tn).n_frames
        for o in c.get('offsets', CFG['offsets']):
            keys = keyframes(N, o)
            gaps = list(zip(keys[:-1], keys[1:]))
            if c['kind'] == 'interp':
                out.append(dict(arm=arm, trial=tn, offset=o))
            elif c['kind'] == 'autoscoper' and c['restart'] == 'first':
                out += [dict(arm=arm, trial=tn, offset=o, bone=n, a=keys[0], b=keys[-1]) for n in CFG['bones']]
            elif c['kind'] == 'autoscoper':
                out += [dict(arm=arm, trial=tn, offset=o, bone=n, a=a, b=b) for n in CFG['bones'] for a, b in gaps]
            else:
                out += [dict(arm=arm, trial=tn, offset=o, a=a, b=b) for a, b in gaps]
    return out


def unit_id(u):
    s = f"{u['trial']}_o{u['offset']}"
    if 'bone' in u: s += f"_{u['bone']}"
    if 'a' in u: s += f"_{u['a']}-{u['b']}"
    return s


def unit_path(u):
    return os.path.join(RESULTS, u['arm'], 'units', unit_id(u) + '.json')


# ------------------------------------------------------------------------------------------- arms
def ref_keys(T, keys, bones):
    return {n: {k: T.ref[n][k] for k in keys} for n in bones}


def spline_track(T, keyposes, kind='spline'):
    out = {}
    for n, kp in keyposes.items():
        Ms = np.full((T.n_frames, 4, 4), np.nan); ks = sorted(kp)
        for k in ks: Ms[k] = kp[k]
        out[n] = interpolate(Ms, ks, np.arange(ks[0], ks[-1] + 1), T.bones[n].pivot, kind)
        out[n] = np.concatenate([np.full((ks[0], 4, 4), np.nan), out[n],
                                 np.full((T.n_frames - ks[-1] - 1, 4, 4), np.nan)])
    return out


def run_interp(u, c):
    T = trial(u['trial']); keys = keyframes(T.n_frames, u['offset'])
    tr = spline_track(T, ref_keys(T, keys, CFG['bones']), c['interp'])
    return {n: {int(f): tr[n][f].tolist() for f in range(keys[0] + 1, keys[-1]) if f not in keys} for n in tr}, {}


def run_autoscoper(u, c, max_frames=None):
    """Forward pass with Autoscoper's 'Previous frame' initial guess: frame f starts from the result for
    f - 1; the first frame starts from the (locked) keyframe a. Each frame: PSO then simplex (Autoscoper
    defaults), on the filtered-NCC cost with the trial's .vie filters."""
    T = trial(u['trial']); n = u['bone']; bone = T.bones[n]
    render = T.render_resolution[0] if c['render'] == 'full' else int(c['render'])
    size = None if c['image'] == 'full' else int(c['image'])
    views = [View(cam, render, T.sample_distance, T.ray_intensity) for cam in T.cams]
    prev = T.ref[n][u['a']]
    poses, meta = {}, {}
    frames = list(range(u['a'] + 1, u['b']))[:max_frames]
    for f in frames:
        P0 = Pose.from_matrix(prev, bone.pivot)
        rads = [T.image(ci, f, size) for ci in range(len(T.cams))]

        def make_f(base_delta, P0=P0, rads=rads):
            P = P0 if base_delta is None else P0.compose(base_delta)

            def fn(x):
                with torch.no_grad():
                    return float(autoscoper_cost(views, rads, bone, P, torch.tensor(x, dtype=torch.float64), T.vie))
            return fn
        t0 = time.time()
        r = autoscoper_optimize(make_f, method='pso', **({'max_epochs': QUICK['epochs']} if QUICK else {}), seed=zlib.crc32(f"{u['trial']}|{u['offset']}|{n}|{f}".encode()))
        P = P0
        for s in r['stages']:
            P = P.compose(s)
        prev = P.matrix(); poses[f] = prev.tolist()
        meta[f] = dict(init=r['init'], pso_cost=r['pso_cost'], cost=r['cost'], epochs=r['epochs'],
                       evals=r['evals'], sec=time.time() - t0)
        log(f'  {unit_id(u)} frame {f}: {r["init"]:.4f} -> PSO {r["pso_cost"]:.4f} -> {r["cost"]:.4f}, '
            f'{r["epochs"]} epochs, {time.time() - t0:.0f}s')
    return {n: poses}, meta


_setup = {}


def setup_anchored(T):
    """Per-trial objects for the proposed method (built once per process)."""
    if T.name not in _setup:
        fovs = fov_masks(T)
        views = {r: [View(cam, r, T.sample_distance, T.ray_intensity) for cam in T.cams] for r in (256, 512)}
        sdfs = {n: collision.SDF(T.bones[n]) for n in CFG['bones']}
        _setup[T.name] = (fovs, views, sdfs)
    return _setup[T.name]


def f2f_for(T, fovs, base, frames):
    crop = []
    for cam in T.cams:
        uv = np.vstack([cam.project(T.bones[n].corners @ base[n][f][:3, :3].T + base[n][f][:3, 3])
                        for n in CFG['bones'] for f in frames])
        lo = np.clip(np.floor(uv.min(0)) - 100, 0, None).astype(int)
        hi = np.minimum(np.ceil(uv.max(0)) + 100, [cam.W, cam.H]).astype(int)
        crop.append((lo[0], lo[1], hi[0], hi[1]))
    return F2F(T, frames, fovs, crop=crop)


def run_anchored(u, c):
    T = trial(u['trial']); keys = keyframes(T.n_frames, u['offset'])
    fovs, views, sdfs = setup_anchored(T)
    base = spline_track(T, ref_keys(T, keys, CFG['bones']))   # spline path through all keyframes of the trial
    a, b = u['a'], u['b']; frames = list(range(a, b + 1))
    f2f = f2f_for(T, fovs, base, frames)
    groups = [CFG['bones']] if c['cofit'] else [[n] for n in CFG['bones']]
    poses, meta = {n: {} for n in CFG['bones']}, {}
    for g in groups:
        fit = AnchoredFit(T, views, fovs, f2f, sdfs, g, [a, b], {(n, f): base[n][f] for n in g for f in frames},
                          cofit=c['cofit'], K=c['K'], w_drr=c['w_drr'], w_f2f=c['w_f2f'], lam=c['lam'],
                          mu=c['mu'], d_min=c['d_min'], key_tol=c.get('key_tol'), res=c['res'], iters=c['iters'],
                          lrs=c['lrs'], batch=c['batch'], seed=a, log=log)
        track, info = fit.run()
        for (n, f), M in track.items():
            if f not in (a, b):
                poses[n][f] = M.tolist()
        info.pop('hist', None); meta['+'.join(g)] = info
    return poses, meta


def run_perframe(u, c):
    T = trial(u['trial']); keys = keyframes(T.n_frames, u['offset'])
    fovs, views, sdfs = setup_anchored(T)
    base = spline_track(T, ref_keys(T, keys, CFG['bones']))
    poses, meta = {n: {} for n in CFG['bones']}, {}
    for f in range(u['a'] + 1, u['b']):
        out, sec = per_frame_fit(T, views, fovs, sdfs, CFG['bones'], f, {(n, f): base[n][f] for n in CFG['bones']},
                                 cofit=c['cofit'], mu=c['mu'], d_min=c['d_min'], res=c['res'], iters=c['iters'],
                                 lrs=c['lrs'])
        for (n, ff), M in out.items(): poses[n][ff] = M.tolist()
        meta[f] = sec
    return poses, meta


def run_unit(u, max_frames=None):
    c = arm_cfg(u['arm']); t0 = time.time()
    if QUICK and 'iters' in c:
        c['iters'] = [QUICK['iters']] * len(c['iters'])
    if c['kind'] == 'interp': poses, meta = run_interp(u, c)
    elif c['kind'] == 'autoscoper': poses, meta = run_autoscoper(u, c, max_frames)
    elif c['kind'] == 'anchored': poses, meta = run_anchored(u, c)
    elif c['kind'] == 'perframe': poses, meta = run_perframe(u, c)
    else: raise ValueError(c['kind'])
    out = dict(unit=u, config=c, poses={n: {str(f): M for f, M in p.items()} for n, p in poses.items()},
               meta={str(k): v for k, v in meta.items()}, sec=time.time() - t0, host=socket.gethostname(),
               device=str(DEVICE), torch=torch.__version__,
               gpu=torch.cuda.get_device_name() if DEVICE.type == 'cuda' else None,
               workers=int(os.environ.get('WORKERS', '1')), threads=torch.get_num_threads(),
               slurm_job=os.environ.get('SLURM_JOB_ID'))
    return out


# ------------------------------------------------------------------------------------------- CLI
def cmd_run(args):
    arms = args.arm.split(',')
    todo = [u for arm in arms for u in units(arm)][args.task::args.ntasks]
    if args.max_units: todo = todo[:args.max_units]
    log(f'device {DEVICE}, task {args.task}/{args.ntasks}: {len(todo)} units')
    for u in todo:
        p = unit_path(u)
        if os.path.exists(p) and not args.quick:
            continue
        log('start', u['arm'], unit_id(u))
        if args.quick:
            QUICK.update(epochs=3, iters=2)
        out = run_unit(u, 2 if args.quick else None)
        if args.quick:                                        # smoke test: don't write partial units
            log('smoke test unit done', u['arm'], unit_id(u), f'{out["sec"]:.0f}s'); continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(out, open(p + '.tmp', 'w')); os.replace(p + '.tmp', p)
        log('done', u['arm'], unit_id(u), f'{out["sec"]:.0f}s')


def cmd_assemble(args):
    arms = args.arm.split(',') if args.arm else list(CFG['arms'])
    for arm in arms:
        by = {}
        for u in units(arm):
            by.setdefault((u['trial'], u['offset']), []).append(u)
        for (tn, o), us in by.items():
            T = trial(tn); tr = {n: np.full((T.n_frames, 4, 4), np.nan) for n in CFG['bones']}
            for k in keyframes(T.n_frames, o):                 # keyframes as given (units may overwrite)
                for n in CFG['bones']: tr[n][k] = T.ref[n][k]
            missing = 0
            for u in us:
                p = unit_path(u)
                if not os.path.exists(p):
                    missing += 1; continue
                for n, d in json.load(open(p))['poses'].items():
                    for f, M in d.items(): tr[n][int(f)] = np.array(M)
            if missing == len(us):
                continue
            path = os.path.join(RESULTS, arm, 'tracks', f'{tn}_o{o}.npz'); os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, **tr)
            log(f'{arm} {tn} offset {o}: {len(us) - missing}/{len(us)} units' + (' (INCOMPLETE)' if missing else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'run', 'assemble'])
    ap.add_argument('--arm', default=None)
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    ap.add_argument('--max-units', type=int, default=0)
    ap.add_argument('--quick', action='store_true',
                    help='smoke test: pso_simplex arms track 2 frames with 3 PSO epochs, new_opt arms run 2 iterations per stage; nothing is saved')
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('THREADS', os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
    if args.cmd == 'list':
        for arm in (args.arm.split(',') if args.arm else list(CFG['arms'])):
            us = units(arm); print(f'# {arm}: {len(us)} units')
            for u in us: print(arm, unit_id(u))
    elif args.cmd == 'run':
        cmd_run(args)
    else:
        cmd_assemble(args)


if __name__ == '__main__':
    main()
