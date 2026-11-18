"""General tracking: keyframe-anchored image refinement and spline interpolation from an Autoscoper trial.

Input: the trial's .cfg, keyframe tracking (.tra, one per bone, NaN on untracked frames) and a keyframe
spacing. Output: optimised and spline tracking as Autoscoper .tra files. Nothing is scored here; compare the
outputs with other tracking in a separate step.

    python general_tracking/run.py keys     --cfg TRIAL.cfg --tra DIR --spacing 10,all   # print the keyframes
    python general_tracking/run.py run      --cfg TRIAL.cfg --tra DIR --spacing 10        # everything, one process
    python general_tracking/run.py assemble --cfg TRIAL.cfg --tra DIR --spacing 10        # write the .tra files

--tra       a folder with one .tra per bone, named after the bone's volume (Femur_keys.tra -> Femur.tif),
            or explicit pairs: --tra Femur=a.tra,Tibia=b.tra
--spacing   N: a keyframe every N frames from the first tracked frame (the first and last frames tracked for
            every bone are always keyframes; a final gap shorter than N/2 joins the one before);
            full: the first and last tracked frames only; all: every tracked frame (sparse keyframes as given)
--kc-tol    keyframe-correction tolerance mm:deg (default config.yaml, 1.5:2); 0:0 holds keyframes as given
--bones, --vie, --out   bone subset; renderer settings if not in the trial folder; results folder

`run` does the keyframe correction, every gap and `assemble` in turn. On the HPC the stages are split
(slurm/submit.sh): `run --stage keys`, then `run --stage gaps --task i --ntasks n` (gaps shared between
workers), then `assemble`. Finished work is skipped, so any step can be rerun to resume.

Outputs in <out>/s<spacing>/ (default <out> = results/general_tracking/<trial>):
    optimised/<trial>_<bone>.tra   keyframe-anchored image refinement (keyframes as corrected)
    spline/<trial>_<bone>.tra      spline through the keyframes as given
    keyframes.json                 keyframe frames, and how far correction moved each keyframe
    diagnostics.csv                per frame and bone: forward/backward chain disagreement (mm, deg), a
                                   reference-free flag for frames to review
    work/                          intermediate results (keyframe correction, one file per gap)
"""
import argparse
import glob
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
from src.geometry import helical_error, pivot_error  # noqa: E402
from src.io import Trial, read_tra, write_tra  # noqa: E402
from src.pipeline import anchored_setup, f2f_for, spline_track  # noqa: E402
from src.tracking import AnchoredFit, KeyframeFit, blend_poses, chain  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, 'general_tracking', 'config.yaml')))
OPT = CFG['optimisation']
QUICK = {}      # smoke test: 2 iterations per stage, nothing saved
S = {}          # set by setup(): trial, bones, keyframe tracking, results folder, keyframe-correction tolerance
MEMO = {}       # corrected keyframes kept in memory (smoke test)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ------------------------------------------------------------------------------------------- inputs
def find_tra(spec, bones):
    """{bone: .tra path} from a folder (one file per bone, matched by name) or 'Bone=path,Bone=path'."""
    if os.path.isdir(spec):
        files = sorted(glob.glob(os.path.join(spec, '*.tra')))
        out = {}
        for n in bones:
            hit = [f for f in files if os.path.basename(f).lower().startswith(n.lower())] or \
                  [f for f in files if n.lower() in os.path.basename(f).lower()]
            if len(hit) != 1:
                raise SystemExit(f'bone {n}: expected one .tra in {spec} named after it, found {len(hit)} '
                                 f'({[os.path.basename(f) for f in hit]}); pass --tra {n}=path explicitly')
            out[n] = hit[0]
        return out
    out = dict(p.split('=', 1) for p in spec.split(','))
    missing = [n for n in bones if n not in out]
    if missing:
        raise SystemExit(f'--tra gives no file for {missing}')
    return {n: out[n] for n in bones}


def setup(args):
    """Load the trial and the keyframe tracking."""
    if not args.cfg or not args.tra:
        raise SystemExit('--cfg and --tra are required')
    cfg = os.path.abspath(args.cfg)
    T = Trial(os.path.dirname(cfg), cfg=cfg, vie=args.vie)
    bones = args.bones.replace(':', ',').split(',') if args.bones else list(T.bones)
    unknown = [n for n in bones if n not in T.bones]
    if unknown:
        raise SystemExit(f'--bones {unknown} not among the volumes in {cfg}: {list(T.bones)}')
    tra = find_tra(args.tra, bones)
    given = {}
    for n, p in tra.items():
        M = read_tra(p)
        if len(M) != T.n_frames:
            raise SystemExit(f'{p}: {len(M)} frames, but the trial has {T.n_frames} radiographs per camera')
        given[n] = M
    tol = [float(x) for x in args.kc_tol.replace(':', ',').split(',')] if args.kc_tol else OPT['keyframe_correction']['tol']
    if len(tol) != 2:
        raise SystemExit(f'--kc-tol needs two values (mm and deg), got {args.kc_tol!r}')
    out = os.path.abspath(args.out or os.environ.get('BVR_RESULTS') or
                          os.path.join(ROOT, 'results', 'general_tracking', T.name))
    S.update(T=T, bones=bones, tra=tra, given=given, tol=tol, out=out, cfg=cfg)
    return T


def tracked_frames():
    """Frames with a keyframe pose for every bone, and frames tracked for some bones only (ignored)."""
    G = S['given']
    ok = np.array([[np.isfinite(G[n][f]).all() for n in S['bones']] for f in range(S['T'].n_frames)])
    return [int(f) for f in np.flatnonzero(ok.all(1))], [int(f) for f in np.flatnonzero(ok.any(1) & ~ok.all(1))]


def keyframes(spacing):
    """Keyframe frames for a spacing. A regular keyframe on a frame not tracked for every bone moves to the
    nearest tracked one."""
    tr, _ = tracked_frames()
    if len(tr) < 2:
        raise SystemExit(f'fewer than two frames are tracked for every bone {S["bones"]}')
    a, b = tr[0], tr[-1]
    if spacing == 'all':
        return tr
    if spacing == 'full':
        keys = [a, b]
    else:
        s = int(spacing)
        keys = list(range(a, b, s))
        if len(keys) > 1 and b - keys[-1] < s / 2:
            keys.pop()
        keys += [b]
    arr, out = np.array(tr), []
    for k in keys:
        k = int(arr[np.argmin(np.abs(arr - k))])
        if not out or k > out[-1]:
            out.append(k)
    return out


def gaps(spacing):
    keys = keyframes(spacing)
    return [(a, b) for a, b in zip(keys[:-1], keys[1:]) if b - a > 1]


def given_keys(keys):
    return {n: {k: S['given'][n][k] for k in keys} for n in S['bones']}


def kc_on():
    return any(t > 0 for t in S['tol'])


def sdir(spacing, *p):
    return os.path.join(S['out'], f's{spacing}', *p)


def keys_path(spacing):
    return sdir(spacing, 'work', 'keyframe_correction.json')


def gap_path(spacing, a, b):
    return sdir(spacing, 'work', 'gaps', f'{a}-{b}.json')


def keyframe_poses(spacing):
    """Keyframes used by the optimisation: corrected (after the keys stage), or as given with --kc-tol 0:0."""
    keys = keyframes(spacing)
    if not kc_on():
        return given_keys(keys)
    if spacing in MEMO:
        return MEMO[spacing]
    p = keys_path(spacing)
    if not os.path.exists(p):
        raise SystemExit(f'corrected keyframes missing ({p}): run --stage keys first')
    d = json.load(open(p))['keyframes']
    return {n: {int(k): np.array(M) for k, M in d[n].items()} for n in S['bones']}


def provenance(t0):
    return dict(cfg=S['cfg'], tra=S['tra'], bones=S['bones'], vie=S['T'].vie_path, sec=time.time() - t0,
                host=socket.gethostname(), device=str(DEVICE), torch=torch.__version__,
                gpu=torch.cuda.get_device_name() if DEVICE.type == 'cuda' else None,
                slurm_job=os.environ.get('SLURM_JOB_ID'))


def iters_of(x):
    return [QUICK['iters']] * len(x) if QUICK else x


# ------------------------------------------------------------------------------------------- method
def correct_keyframes(spacing):
    """Re-fit each keyframe to the images of frames k - window .. k + window (bones co-fitted), within tol."""
    T = S['T']; BONES = S['bones']; kc = OPT['keyframe_correction']; keys = keyframes(spacing); t0 = time.time()
    given = given_keys(keys)
    fovs, views, sdfs = anchored_setup(T, BONES)
    spline = spline_track(T, given)
    w = int(kc['window']); out = {n: {} for n in BONES}; meta = {}
    for k in keys:
        frames = list(range(max(0, k - w), min(T.n_frames - 1, k + w) + 1))
        base = {(n, f): spline[n][f] if np.isfinite(spline[n][f]).all() else given[n][k] for n in BONES for f in frames}
        f2f = f2f_for(T, fovs, {n: {f: base[(n, f)] for f in frames} for n in BONES}, frames, BONES, margin=150)
        fit = KeyframeFit(T, views, fovs, f2f, sdfs, BONES, k, frames, base, tol=S['tol'], w_drr=OPT['w_drr'],
                          w_f2f=OPT['w_f2f'], mu=OPT['mu'], d_min=OPT['d_min'], key_w=kc['key_w'], res=kc['res'],
                          iters=iters_of(kc['iters']), lrs=kc['lrs'], step=kc.get('step', 'tol'), log=log)
        poses, info = fit.run()
        for n in BONES: out[n][k] = poses[n]
        meta[str(k)] = info
    return out, dict(meta=meta, tol=S['tol'], config=kc, **provenance(t0))


def fit_gap(spacing, a, b):
    """Stepwise chains from keyframes a and b, blended by distance, then the anchored whole-gap refinement."""
    T = S['T']; BONES = S['bones']; c = OPT; t0 = time.time()
    fovs, views, sdfs = anchored_setup(T, BONES)
    K = keyframe_poses(spacing)
    spline = spline_track(T, K)
    inner = list(range(a + 1, b)); frames = list(range(a, b + 1))
    f2f = f2f_for(T, fovs, spline, frames, BONES, margin=300)
    pts = {n: torch.tensor(T.bones[n].surface_points(2500, 0.6, seed=0)) for n in BONES}
    kw = dict(w_drr=c['w_drr'], w_f2f=c['w_f2f'], mu=c['mu'], d_min=c['d_min'], res=c['step_res'],
              iters=iters_of(c['step_iters']), lrs=c['step_lrs'])
    fwd = chain(T, views, fovs, f2f, sdfs, BONES, {n: K[n][a] for n in BONES}, inner, +1,
                {n: spline[n][a + 1] for n in BONES}, pts, **kw)
    bwd = chain(T, views, fovs, f2f, sdfs, BONES, {n: K[n][b] for n in BONES}, inner[::-1], -1,
                {n: spline[n][b - 1] for n in BONES}, pts, **kw)
    t_chain = time.time() - t0
    base = {n: spline[n].copy() for n in BONES}; disagree = {n: {} for n in BONES}
    for n in BONES:
        piv = T.bones[n].pivot
        for f in inner:
            Mf, Mb = fwd[(n, f)], bwd[(n, f)]
            base[n][f] = blend_poses(Mf, Mb, (f - a) / (b - a), piv)
            dR = np.degrees(np.arccos(np.clip((np.trace(Mf[:3, :3].T @ Mb[:3, :3]) - 1) / 2, -1, 1)))
            disagree[n][str(f)] = [float(np.linalg.norm((Mf[:3, :3] - Mb[:3, :3]) @ piv + Mf[:3, 3] - Mb[:3, 3])), float(dR)]
    log(f'    chains {a}-{b}: {t_chain:.0f}s, max disagreement {max(v[0] for d in disagree.values() for v in d.values()):.2f} mm')
    gap = b - a
    Kb = max(1, math.ceil(c['K_per_10'] * gap / 10))
    batch = max(1, min(gap - 1, math.ceil(c['batch_per_10'] * gap / 10)))
    fit = AnchoredFit(T, views, fovs, f2f, sdfs, BONES, [a, b], {(n, f): base[n][f] for n in BONES for f in frames},
                      cofit=c['cofit'], K=Kb, w_drr=c['w_drr'], w_f2f=c['w_f2f'], lam=c['lam'], mu=c['mu'],
                      d_min=c['d_min'], res=c['res'], iters=iters_of(c['iters']), lrs=c['lrs'], batch=batch, seed=a, log=log)
    track, info = fit.run()
    info.pop('hist', None)
    poses = {n: {str(f): track[(n, f)].tolist() for f in inner} for n in BONES}
    return dict(gap=[a, b], poses=poses, chain_disagreement=disagree, chain_sec=t_chain, K=Kb, batch=batch,
                fit=info, config=c, **provenance(t0))


def save(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(d, open(path + '.tmp', 'w')); os.replace(path + '.tmp', path)


# ------------------------------------------------------------------------------------------- commands
def stage_keys(spacing, quick):
    if not kc_on():
        return
    p = keys_path(spacing)
    if os.path.exists(p) and not quick:
        return
    log(f'start keyframe correction, spacing {spacing}: {len(keyframes(spacing))} keyframes, tol {S["tol"]}')
    kp, meta = correct_keyframes(spacing)
    if quick:
        MEMO[spacing] = kp; log('smoke test: keyframe correction done'); return
    save(p, dict(keyframes={n: {str(k): M.tolist() for k, M in d.items()} for n, d in kp.items()}, **meta))
    log(f'done keyframe correction, spacing {spacing}: {meta["sec"]:.0f}s')


def stage_gaps(spacing, task, ntasks, quick):
    todo = gaps(spacing)[task::ntasks]
    log(f'spacing {spacing}: {len(todo)} gaps for task {task}/{ntasks}')
    for a, b in todo:
        p = gap_path(spacing, a, b)
        if os.path.exists(p) and not quick:
            continue
        log(f'start gap {a}-{b}, spacing {spacing}')
        d = fit_gap(spacing, a, b)
        if quick:
            log(f'smoke test: gap {a}-{b} done, {d["sec"]:.0f}s'); continue
        save(p, d); log(f'done gap {a}-{b}, spacing {spacing}: {d["sec"]:.0f}s')


def assemble(spacing):
    """Write the optimised and spline .tra files, keyframes.json and diagnostics.csv for one spacing."""
    T = S['T']; BONES = S['bones']; keys = keyframes(spacing); gs = gaps(spacing)
    given = given_keys(keys)
    spline = spline_track(T, given, CFG['spline'])
    os.makedirs(sdir(spacing, 'spline'), exist_ok=True)
    for n in BONES:
        write_tra(sdir(spacing, 'spline', f'{T.name}_{n}.tra'), spline[n])
    K = keyframe_poses(spacing)
    opt = {n: np.full((T.n_frames, 4, 4), np.nan) for n in BONES}
    for n in BONES:
        for k, M in K[n].items(): opt[n][k] = M
    done, diag = 0, []
    for a, b in gs:
        p = gap_path(spacing, a, b)
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); done += 1
        for n in BONES:
            for f, M in d['poses'][n].items(): opt[n][int(f)] = np.array(M)
            for f, (mm, deg) in d['chain_disagreement'][n].items():
                diag.append((int(f), n, a, b, mm, deg))
    os.makedirs(sdir(spacing, 'optimised'), exist_ok=True)
    for n in BONES:
        write_tra(sdir(spacing, 'optimised', f'{T.name}_{n}.tra'), opt[n])
    with open(sdir(spacing, 'diagnostics.csv'), 'w') as fh:
        fh.write('frame,bone,gap_start,gap_end,chain_disagreement_mm,chain_disagreement_deg\n')
        for r in sorted(diag):
            fh.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.4f},{r[5]:.4f}\n')
    moved = {n: {str(k): dict(pivot_mm=pivot_error(K[n][k], given[n][k], T.bones[n].pivot),
                              rot_deg=helical_error(K[n][k], given[n][k])[0]) for k in keys} for n in BONES}
    json.dump(dict(trial=T.name, cfg=S['cfg'], tra=S['tra'], bones=S['bones'], spacing=spacing, keyframes=keys,
                   keyframe_correction_tol=S['tol'] if kc_on() else None, correction_moved=moved,
                   gaps_done=done, gaps_total=len(gs)),
              open(sdir(spacing, 'keyframes.json'), 'w'), indent=1)
    status = 'complete' if done == len(gs) else f'INCOMPLETE: {len(gs) - done} gaps missing (NaN in optimised/)'
    log(f'spacing {spacing}: {done}/{len(gs)} gaps, {status}; written to {sdir(spacing)}')


def cmd_keys(spacings):
    T = S['T']; tr, partial = tracked_frames()
    print(f'{T.name}: {T.n_frames} frames; bones {S["bones"]}; keyframe tracking from '
          + ', '.join(os.path.basename(p) for p in S['tra'].values()))
    print(f'  tracked for every bone: {len(tr)} frames ({tr[0]}-{tr[-1]})' if tr else '  no frame tracked for every bone')
    if partial:
        print(f'  ignored: {len(partial)} frames tracked for some bones only')
    for s in spacings:
        k = keyframes(s); g = np.diff(k)
        print(f'  spacing {s}: {len(k)} keyframes, gaps {sorted(set(int(x) for x in g))}, '
              f'{int((g - 1).sum())} frames to fill: {k if len(k) <= 30 else str(k[:30])[:-1] + ", ...]"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['keys', 'run', 'assemble', 'ngaps'])
    ap.add_argument('--cfg', help='Autoscoper .cfg of the trial (paths in it are relative to its folder)')
    ap.add_argument('--tra', help='folder with one keyframe .tra per bone, or Bone=path,Bone=path')
    ap.add_argument('--spacing', required=True, help='N, full or all; several comma-separated')
    ap.add_argument('--stage', default='all', choices=['all', 'keys', 'gaps'])
    ap.add_argument('--task', type=int, default=0)
    ap.add_argument('--ntasks', type=int, default=1)
    ap.add_argument('--kc-tol', default=None, help='keyframe-correction tolerance mm:deg; 0:0 = no correction')
    ap.add_argument('--bones', default=None, help='subset of the .cfg volumes, comma- or colon-separated')
    ap.add_argument('--vie', default=None, help='Autoscoper renderer settings, if not found in the trial folder')
    ap.add_argument('--out', default=None, help='results folder (default results/general_tracking/<trial>)')
    ap.add_argument('--quick', action='store_true', help='smoke test: 2 iterations per stage; nothing saved')
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('THREADS', os.environ.get('SLURM_CPUS_PER_TASK', '2'))))
    setup(args)
    spacings = args.spacing.split(',')
    if args.quick: QUICK.update(iters=2)
    if args.cmd == 'keys':
        cmd_keys(spacings)
    elif args.cmd == 'ngaps':
        print(sum(len(gaps(s)) for s in spacings))
    elif args.cmd == 'assemble':
        for s in spacings: assemble(s)
    else:
        log(f'device {DEVICE}, trial {S["T"].name}, bones {S["bones"]}, spacing {args.spacing}, stage {args.stage}')
        for s in spacings:
            if args.stage in ('all', 'keys') and args.task == 0:
                stage_keys(s, args.quick)
            if args.stage in ('all', 'gaps'):
                stage_gaps(s, args.task, args.ntasks, args.quick)
            if args.stage == 'all' and args.ntasks == 1 and not args.quick:
                assemble(s)


if __name__ == '__main__':
    main()
