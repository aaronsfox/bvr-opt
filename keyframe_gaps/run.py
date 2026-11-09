"""Keyframe gaps runner: one arm at one or more keyframe spacings, split into independent units.

    python keyframe_gaps/run.py list     [--arm new_opt] [--spacing 5,10,full]
    python keyframe_gaps/run.py keys                                   # print the keyframes per trial and spacing
    python keyframe_gaps/run.py run      --arm new_opt --spacing 20 [--task 0 --ntasks 1] [--quick]
    python keyframe_gaps/run.py assemble [--arm new_opt] [--spacing 20]

Units: new_opt, new_opt_stepwise -> (trial, spacing, gap), all bones co-fitted; keyframe_interp -> (trial, spacing).
new_opt_stepwise also writes its chains-only path (before refinement) as the pseudo-arm new_opt_stepwise_chains
at `assemble`.
`run` processes units[task::ntasks] and skips units already written, so jobs can be resubmitted.
Outputs ($BVR_RESULTS, default <repo>/results/keyframe_gaps), one folder per arm and spacing:
    <arm>/s<spacing>/units/<trial>_<a>-<b>.json      poses, settings, timings
    <arm>/s<spacing>/tracks/<trial>.npz               after `assemble`: {bone: (n_frames, 4, 4)}
"""
import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch  # noqa: E402

from src.device import DEVICE  # noqa: E402
from src.io import Trial  # noqa: E402
from src.pipeline import anchored_setup, f2f_for, ref_keys, spline_track  # noqa: E402
from src.tracking import AnchoredFit, KeyframeFit, blend_poses, chain  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, 'keyframe_gaps', 'config.yaml')))
RESULTS = os.environ.get('BVR_RESULTS', os.path.join(ROOT, 'results', 'keyframe_gaps'))
BONES = CFG['bones']
QUICK = {}   # smoke-test overrides (never used for real runs)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


_trials = {}


def trial(name):
    if name not in _trials:
        _trials[name] = Trial(os.path.join(ROOT, CFG['data_root'], name))
    return _trials[name]


def keyframes(n_frames, spacing):
    """Bookends at `bookend` and n_frames - 1 - `bookend`; regular keyframes every `spacing` frames between
    them; a final gap shorter than spacing / 2 is merged into the previous one."""
    a, b = CFG['bookend'], n_frames - 1 - CFG['bookend']
    if str(spacing) == 'full':
        return [a, b]
    s = int(spacing)
    keys = list(range(a, b, s))
    if len(keys) > 1 and b - keys[-1] < s / 2:
        keys.pop()
    return keys + [b]


def spacings(arg):
    return [str(s) for s in (arg.split(',') if arg else CFG['spacings'])]


def units(arm, spacing, stage='gaps'):
    """stage 'gaps': the arm's units; stage 'keys': keyframe-correction units (one per trial) for arms with
    keyframe_correction, which must finish before the gap units of that spacing."""
    c = CFG['arms'][arm]; out = []
    if stage == 'keys':
        return [dict(arm=arm, spacing=str(spacing), trial=tn, stage='keys') for tn in CFG['trials']] \
            if 'keyframe_correction' in c else []
    for tn in CFG['trials']:
        keys = keyframes(trial(tn).n_frames, spacing)
        if c['kind'] == 'interp':
            out.append(dict(arm=arm, spacing=str(spacing), trial=tn, a=keys[0], b=keys[-1]))
        else:
            out += [dict(arm=arm, spacing=str(spacing), trial=tn, a=a, b=b) for a, b in zip(keys[:-1], keys[1:])]
    return out


def unit_id(u):
    return u['trial'] if u.get('stage') == 'keys' else f"{u['trial']}_{u['a']}-{u['b']}"


def arm_dir(arm, spacing):
    return os.path.join(RESULTS, arm, f's{spacing}')


def unit_path(u):
    sub = 'keyframes' if u.get('stage') == 'keys' else 'units'
    return os.path.join(arm_dir(u['arm'], u['spacing']), sub, unit_id(u) + '.json')


def given_keyframes(T, keys):
    """Keyframe poses as a rater would supply them: {bone: {frame: 4x4}}. Perfect rater for now."""
    if CFG['keyframes'] == 'reference':
        return ref_keys(T, keys, BONES)
    raise ValueError(CFG['keyframes'])


def keyframe_poses(arm, spacing, T):
    """Keyframes used by an arm: the given keyframes, or the corrected ones for arms with keyframe_correction."""
    keys = keyframes(T.n_frames, spacing)
    if 'keyframe_correction' not in CFG['arms'][arm]:
        return given_keyframes(T, keys)
    p = unit_path(dict(arm=arm, spacing=str(spacing), trial=T.name, stage='keys'))
    if not os.path.exists(p):
        raise FileNotFoundError(f'corrected keyframes missing ({p}): run the keys stage for {arm} spacing {spacing} first')
    d = json.load(open(p))['keyframes']
    return {n: {int(k): np.array(M) for k, M in d[n].items()} for n in BONES}


def run_keys(u, c):
    """Correct every given keyframe of one trial at one spacing (keyframe_correction settings)."""
    T = trial(u['trial']); kc = c['keyframe_correction']; keys = keyframes(T.n_frames, u['spacing'])
    given = given_keyframes(T, keys)
    fovs, views, sdfs = anchored_setup(T, BONES)
    spline = spline_track(T, given)
    w = int(kc['window']); out = {n: {} for n in BONES}; meta = {}
    iters = [QUICK['iters']] * len(kc['iters']) if QUICK else kc['iters']
    for k in keys:
        frames = list(range(max(0, k - w), min(T.n_frames - 1, k + w) + 1))
        base = {}
        for n in BONES:
            for f in frames:                             # outside the bookends: hold the keyframe pose
                base[(n, f)] = spline[n][f] if np.isfinite(spline[n][f]).all() else given[n][k]
        f2f = f2f_for(T, fovs, {n: {f: base[(n, f)] for f in frames} for n in BONES}, frames, BONES, margin=150)
        fit = KeyframeFit(T, views, fovs, f2f, sdfs, BONES, k, frames, base, tol=kc['tol'], w_drr=c['w_drr'],
                          w_f2f=c['w_f2f'], mu=c['mu'], d_min=c['d_min'], key_w=kc['key_w'], res=kc['res'],
                          iters=iters, lrs=kc['lrs'], log=log)
        poses, info = fit.run()
        for n in BONES: out[n][k] = poses[n].tolist()
        meta[k] = info
    return out, meta


# ------------------------------------------------------------------------------------------- arms
def run_interp(u, c):
    T = trial(u['trial']); keys = keyframes(T.n_frames, u['spacing'])
    tr = spline_track(T, keyframe_poses(u['arm'], u['spacing'], T), c['interp'])
    return {n: {f: tr[n][f].tolist() for f in range(keys[0] + 1, keys[-1]) if f not in keys} for n in BONES}, {}


def run_anchored(u, c, base=None, f2f=None):
    """Anchored fit of one gap, all bones co-fitted. Base path: the spline through all keyframes of the trial,
    or `base` {bone: (n_frames, 4, 4)} when given (the stepwise arm)."""
    T = trial(u['trial']); keys = keyframes(T.n_frames, u['spacing'])
    fovs, views, sdfs = anchored_setup(T, BONES)
    if base is None:
        base = spline_track(T, keyframe_poses(u['arm'], u['spacing'], T))
    a, b = u['a'], u['b']; frames = list(range(a, b + 1)); gap = b - a
    K = max(1, math.ceil(c['K_per_10'] * gap / 10))
    batch = max(1, min(gap - 1, math.ceil(c['batch_per_10'] * gap / 10)))
    iters = [QUICK['iters']] * len(c['iters']) if QUICK else c['iters']
    fit = AnchoredFit(T, views, fovs, f2f or f2f_for(T, fovs, base, frames, BONES), sdfs, BONES, [a, b],
                      {(n, f): base[n][f] for n in BONES for f in frames}, cofit=c['cofit'], K=K,
                      w_drr=c['w_drr'], w_f2f=c['w_f2f'], lam=c['lam'], mu=c['mu'], d_min=c['d_min'],
                      res=c['res'], iters=iters, lrs=c['lrs'], batch=batch, seed=a, log=log)
    track, info = fit.run()
    info.pop('hist', None)
    poses = {n: {f: track[(n, f)].tolist() for f in range(a + 1, b)} for n in BONES}
    return poses, dict(K=K, batch=batch, gap=gap, iters=iters, **info)


def run_stepwise(u, c):
    """Forward chain from keyframe a and backward chain from keyframe b, blended by distance (weight on the
    backward chain = (f - a) / (b - a)), then the anchored refinement starting from the blended path."""
    T = trial(u['trial']); keys = keyframes(T.n_frames, u['spacing'])
    fovs, views, sdfs = anchored_setup(T, BONES)
    K = keyframe_poses(u['arm'], u['spacing'], T)             # given, or corrected (keyframe_correction)
    spline = spline_track(T, K)
    a, b = u['a'], u['b']; inner = list(range(a + 1, b))
    f2f = f2f_for(T, fovs, spline, list(range(a, b + 1)), BONES, margin=300)   # wide: the true path may be far from the spline
    pts = {n: torch.tensor(T.bones[n].surface_points(2500, 0.6, seed=0)) for n in BONES}
    step_iters = [QUICK['iters']] * len(c['step_iters']) if QUICK else c['step_iters']
    kw = dict(w_drr=c['w_drr'], w_f2f=c['w_f2f'], mu=c['mu'], d_min=c['d_min'], res=c['step_res'],
              iters=step_iters, lrs=c['step_lrs'])
    t0 = time.time()
    fwd = chain(T, views, fovs, f2f, sdfs, BONES, {n: K[n][a] for n in BONES}, inner, +1,
                {n: spline[n][a + 1] for n in BONES}, pts, **kw)
    bwd = chain(T, views, fovs, f2f, sdfs, BONES, {n: K[n][b] for n in BONES}, inner[::-1], -1,
                {n: spline[n][b - 1] for n in BONES}, pts, **kw)
    t_chain = time.time() - t0
    base = {n: spline[n].copy() for n in BONES}; blend, disagree = {n: {} for n in BONES}, {n: {} for n in BONES}
    for n in BONES:
        piv = T.bones[n].pivot
        for f in inner:
            Mf, Mb = fwd[(n, f)], bwd[(n, f)]
            base[n][f] = blend_poses(Mf, Mb, (f - a) / (b - a), piv); blend[n][f] = base[n][f].tolist()
            dR = np.degrees(np.arccos(np.clip((np.trace(Mf[:3, :3].T @ Mb[:3, :3]) - 1) / 2, -1, 1)))
            disagree[n][f] = [float(np.linalg.norm((Mf[:3, :3] - Mb[:3, :3]) @ piv + Mf[:3, 3] - Mb[:3, 3])), float(dR)]
    log(f'    chains {a}-{b}: {t_chain:.0f}s, max disagreement '
        f'{max(v[0] for d in disagree.values() for v in d.values()):.2f} mm')
    poses, meta = run_anchored(u, c, base=base, f2f=f2f)
    meta.update(chain_sec=t_chain, step_iters=step_iters, blend=blend, chain_disagreement=disagree)
    return poses, meta


def run_unit(u):
    c = CFG['arms'][u['arm']]; t0 = time.time()
    if u.get('stage') == 'keys':
        kp, meta = run_keys(u, c)
        return dict(unit=u, config=c, keyframes={n: {str(k): M for k, M in p.items()} for n, p in kp.items()},
                    meta={str(k): v for k, v in meta.items()}, sec=time.time() - t0, host=socket.gethostname(),
                    device=str(DEVICE), gpu=torch.cuda.get_device_name() if DEVICE.type == 'cuda' else None)
    poses, meta = {'interp': run_interp, 'anchored': run_anchored, 'stepwise': run_stepwise}[c['kind']](u, c)
    return dict(unit=u, config=c, poses={n: {str(f): M for f, M in p.items()} for n, p in poses.items()},
                meta=meta, sec=time.time() - t0, host=socket.gethostname(), device=str(DEVICE),
                torch=torch.__version__, gpu=torch.cuda.get_device_name() if DEVICE.type == 'cuda' else None,
                workers=int(os.environ.get('WORKERS', '1')), threads=torch.get_num_threads(),
                slurm_job=os.environ.get('SLURM_JOB_ID'))


# ------------------------------------------------------------------------------------------- CLI
def cmd_run(args):
    todo = [u for s in spacings(args.spacing) for u in units(args.arm, s, args.stage)][args.task::args.ntasks]
    if args.max_units: todo = todo[:args.max_units]
    log(f'device {DEVICE}, {args.arm}, spacing {args.spacing}, stage {args.stage}, task {args.task}/{args.ntasks}: {len(todo)} units')
    if args.quick: QUICK.update(iters=2)
    for u in todo:
        p = unit_path(u)
        if os.path.exists(p) and not args.quick:
            continue
        log('start', u['arm'], f"s{u['spacing']}", unit_id(u))
        out = run_unit(u)
        if args.quick:
            log('smoke test unit done', u['arm'], f"s{u['spacing']}", unit_id(u), f'{out["sec"]:.0f}s'); continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(out, open(p + '.tmp', 'w')); os.replace(p + '.tmp', p)
        log('done', u['arm'], f"s{u['spacing']}", unit_id(u), f'{out["sec"]:.0f}s')


def cmd_assemble(args):
    for arm in (args.arm.split(',') if args.arm else list(CFG['arms'])):
        for s in spacings(args.spacing):
            by = {}
            for u in units(arm, s): by.setdefault(u['trial'], []).append(u)
            for tn, us in by.items():
                T = trial(tn); tr = {n: np.full((T.n_frames, 4, 4), np.nan) for n in BONES}
                done = [u for u in us if os.path.exists(unit_path(u))]
                if not done:
                    continue
                K = keyframe_poses(arm, s, T)                 # keyframes as used (corrected for _kc arms)
                for n in BONES:
                    for k, M in K[n].items(): tr[n][k] = M
                bl = {n: tr[n].copy() for n in BONES}; has_blend = False
                for u in done:
                    d = json.load(open(unit_path(u)))
                    for n, p in d['poses'].items():
                        for f, M in p.items(): tr[n][int(f)] = np.array(M)
                    if 'blend' in d['meta']:                  # stepwise arm: also keep the chains-only path
                        has_blend = True
                        for n, p in d['meta']['blend'].items():
                            for f, M in p.items(): bl[n][int(f)] = np.array(M)
                path = os.path.join(arm_dir(arm, s), 'tracks', f'{tn}.npz'); os.makedirs(os.path.dirname(path), exist_ok=True)
                np.savez(path, **tr)
                if has_blend:
                    path = os.path.join(arm_dir(arm + '_chains', s), 'tracks', f'{tn}.npz')
                    os.makedirs(os.path.dirname(path), exist_ok=True); np.savez(path, **bl)
                log(f'{arm} s{s} {tn}: {len(done)}/{len(us)} units' + (' (INCOMPLETE)' if len(done) < len(us) else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['list', 'keys', 'run', 'assemble', 'has-kc'])
    ap.add_argument('--stage', default='gaps', choices=['gaps', 'keys'],
                    help='keys: keyframe-correction stage (arms with keyframe_correction), run before the gaps')
    ap.add_argument('--arm', default=None)
    ap.add_argument('--spacing', default=None, help='comma-separated, e.g. 5,10,full (default: all)')
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    ap.add_argument('--max-units', type=int, default=0)
    ap.add_argument('--quick', action='store_true', help='smoke test: 2 iterations per stage; nothing is saved')
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('THREADS', os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
    if args.cmd == 'has-kc':
        print('yes' if 'keyframe_correction' in CFG['arms'][args.arm] else 'no'); return
    if args.cmd == 'keys':
        for tn in CFG['trials']:
            N = trial(tn).n_frames
            for s in spacings(args.spacing):
                print(f'{tn} ({N} frames) spacing {s}: {keyframes(N, s)}')
    elif args.cmd == 'list':
        for arm in (args.arm.split(',') if args.arm else list(CFG['arms'])):
            for s in spacings(args.spacing):
                ks = units(arm, s, 'keys')
                if ks: print(f'# {arm} spacing {s}: {len(ks)} keyframe-correction units (stage keys)')
                us = units(arm, s); print(f'# {arm} spacing {s}: {len(us)} units')
                for u in us: print(arm, f's{s}', unit_id(u))
    elif args.cmd == 'run':
        if not args.arm: raise SystemExit('--arm is required')
        cmd_run(args)
    else:
        cmd_assemble(args)


if __name__ == '__main__':
    main()
