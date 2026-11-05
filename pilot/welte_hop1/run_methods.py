"""Pilot of Analysis 1 on T0014_hop108_0001, window frames 20-40 (keyframes 20, 30, 40).

Keyframes are the reference (bead-based) poses every 10 frames over the whole trial (0, 10, ..., 70), i.e. a
perfect rater. Frames between keyframes are never shown to any method; they are only used for scoring.

Arms (results/<arm>.npz: {bone: (n_frames, 4, 4)}, NaN where an arm produced no pose):
  S1_spline, S1_linear       interpolation between keyframes (quaternion spline / slerp), raw and Butterworth 10 Hz
  S2                         each keyframe 10..50 refined by Autoscoper simplex (filtered NCC, 512^2), then spline (+BW)
  S3                         per-frame Autoscoper PSO + simplex from the spline pose (filtered NCC, 512^2)
  P1                         anchored gap fit, bones fitted alone
  P2                         anchored gap fit, tibia + talus + calcaneus co-fitted
  P3, P3tol                  P2 + contact penalty (d_min = 0 mm; tolerant d_min = -2 mm)
  P2_drr, P2_f2f, PF         components: DRR term only, frame-to-frame only, no anchoring (per-frame fit)

Usage: python run_methods.py <arm> [frames for S3]
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from bvr_opt import collision
from bvr_opt.costs import F2F, autoscoper_cost, fov_masks
from bvr_opt.drr import View
from bvr_opt.geometry import Pose, butterworth, interpolate
from bvr_opt.io import Trial
from bvr_opt.optim import autoscoper_optimize
from bvr_opt.tracking import GapFit, per_frame_fit

torch.set_num_threads(int(os.environ.get('THREADS', '2')))
TRIAL = os.path.join(ROOT, 'datasets/Welte2022/data/SOL001A/T0014_hop108_0001')
RES = os.path.join(HERE, 'results'); os.makedirs(RES, exist_ok=True)
SPACING, WIN = 10, (20, 40)
GAPS = [(20, 30), (30, 40)]
BONES = ['tib', 'tal', 'cal']

T = Trial(TRIAL)
N = T.n_frames
KEYS = np.unique(np.r_[np.arange(0, N, SPACING), N - 1])   # 0, 10, ..., 70 and the last frame (73)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def save(arm, track, meta=None):
    np.savez(os.path.join(RES, f'{arm}.npz'), **{n: track[n] for n in track})
    if meta is not None:
        json.dump(meta, open(os.path.join(RES, f'{arm}.json'), 'w'), indent=1)
    log('saved', arm)


def spline_track(keyposes, kind='spline'):
    """keyposes: {bone: {frame: 4x4}} -> {bone: (N, 4, 4)} interpolated over the whole trial."""
    out = {}
    for n in BONES:
        Ms = np.full((N, 4, 4), np.nan); ks = sorted(keyposes[n])
        for k in ks: Ms[k] = keyposes[n][k]
        out[n] = interpolate(Ms, ks, np.arange(N), T.bones[n].pivot, kind)
    return out


def bw(track):
    return {n: butterworth(track[n], T.bones[n].pivot, fs=250.0, fc=10.0, order=2) for n in track}


def ref_keys():
    return {n: {int(k): T.ref[n][k] for k in KEYS} for n in BONES}


def autoscoper_views(res=512):
    return [View(c, res, T.sample_distance, T.ray_intensity) for c in T.cams]


def make_cost(views, bone, f, size, P0):
    """Autoscoper cost as a function of an increment, for composition of stages (Tracker::optimize)."""
    rads = [T.image(ci, f, size) for ci in range(len(T.cams))]
    def make_f(base_delta):
        P = P0 if base_delta is None else P0.compose(base_delta)
        def fn(x):
            with torch.no_grad():
                return float(autoscoper_cost(views, rads, bone, P, torch.tensor(x, dtype=torch.float64), T.vie))
        return fn
    return make_f


def compose_stages(P0, stages):
    P = P0
    for s in stages:
        P = P.compose(s)
    return P.matrix()


# ------------------------------------------------------------------------------------------- arms
def arm_s1():
    for kind in ('spline', 'linear'):
        tr = spline_track(ref_keys(), kind)
        save(f'S1_{kind}_raw', tr); save(f'S1_{kind}', bw(tr))


def arm_s2(size=512):
    views = autoscoper_views(size)
    keys = ref_keys(); meta = {}
    for n in BONES:
        for k in [10, 20, 30, 40, 50]:
            P0 = Pose.from_matrix(T.ref[n][k], T.bones[n].pivot)
            t0 = time.time()
            r = autoscoper_optimize(make_cost(views, T.bones[n], k, size, P0), method='simplex')
            keys[n][k] = compose_stages(P0, r['stages'])
            meta[f'{n}_{k}'] = dict(r, stages=[s.tolist() for s in r['stages']], sec=time.time() - t0)
            log(f'S2 {n} keyframe {k}: cost {r["init"]:.4f} -> {r["cost"]:.4f}, {r["evals"]} evals, {time.time() - t0:.0f}s')
    tr = spline_track(keys)
    save('S2_raw', tr, meta); save('S2', bw(tr))


def arm_s3(frames, size=512, tag='S3'):
    views = autoscoper_views(size)
    base = spline_track(ref_keys())
    path = os.path.join(RES, f'{tag}_frames.json')
    done = json.load(open(path)) if os.path.exists(path) else {}
    for f in frames:
        for n in BONES:
            key = f'{n}_{f}'
            if key in done:
                continue
            P0 = Pose.from_matrix(base[n][f], T.bones[n].pivot)
            t0 = time.time()
            r = autoscoper_optimize(make_cost(views, T.bones[n], f, size, P0), method='pso', seed=1000 * f + BONES.index(n))
            M = compose_stages(P0, r['stages'])
            done[key] = dict(M=M.tolist(), cost=r['cost'], init=r['init'], pso_cost=r['pso_cost'], evals=r['evals'],
                             epochs=r['epochs'], sec=time.time() - t0)
            json.dump(done, open(path, 'w'))
            log(f'{tag} {n} frame {f}: cost {r["init"]:.4f} -> PSO {r["pso_cost"]:.4f} -> {r["cost"]:.4f}, '
                f'{r["epochs"]} epochs, {r["evals"]} evals, {time.time() - t0:.0f}s')
    tr = {n: np.full((N, 4, 4), np.nan) for n in BONES}
    for key, v in done.items():
        n, f = key.split('_'); tr[n][int(f)] = np.array(v['M'])
    save(tag, tr)


def setup_p():
    fovs = fov_masks(T)
    views = {r: [View(c, r, T.sample_distance, T.ray_intensity) for c in T.cams] for r in (256, 512)}
    sdfs = {n: collision.SDF(T.bones[n]) for n in BONES}
    base = spline_track(ref_keys())                  # hard anchors at keyframes, spline path between
    crop = []                                        # image region the bones occupy over the window (+100 px)
    for ci, cam in enumerate(T.cams):
        uv = np.vstack([cam.project(T.bones[n].corners @ base[n][f][:3, :3].T + base[n][f][:3, 3])
                        for n in BONES for f in range(WIN[0], WIN[1] + 1)])
        lo = np.clip(np.floor(uv.min(0)) - 100, 0, None).astype(int); hi = np.minimum(np.ceil(uv.max(0)) + 100, [cam.W, cam.H]).astype(int)
        crop.append((lo[0], lo[1], hi[0], hi[1]))
    f2f = F2F(T, range(WIN[0], WIN[1] + 1), fovs, crop=crop)
    return fovs, views, f2f, sdfs, base


def arm_p(arm, cofit=True, w_drr=1.0, w_f2f=2.0, mu=0.0, d_min=0.0):
    fovs, views, f2f, sdfs, base = setup_p()
    tr = {n: np.full((N, 4, 4), np.nan) for n in BONES}
    for n in BONES:
        for k in (WIN[0], 30, WIN[1]): tr[n][k] = T.ref[n][k]
    meta = {}
    groups = [BONES] if cofit else [[n] for n in BONES]
    for a, b in GAPS:
        for g in groups:
            fit = GapFit(T, views, fovs, f2f, sdfs, g, a, b, {(n, f): base[n][f] for n in g for f in range(a, b + 1)},
                         cofit=cofit, w_drr=w_drr, w_f2f=w_f2f, mu=mu, d_min=d_min, seed=a, log=log)
            track, info = fit.run()
            for (n, f), M in track.items(): tr[n][f] = M
            meta[f'{"+".join(g)}_{a}_{b}'] = info
    save(arm, tr, meta)


def arm_pf(arm, mu=0.0, d_min=0.0):
    fovs, views, f2f, sdfs, base = setup_p()
    tr = {n: np.full((N, 4, 4), np.nan) for n in BONES}
    meta = {}
    for f in range(WIN[0] + 1, WIN[1]):
        if f % SPACING == 0:
            continue
        out, sec = per_frame_fit(T, views, fovs, sdfs, BONES, f, {(n, f): base[n][f] for n in BONES}, mu=mu, d_min=d_min)
        for (n, ff), M in out.items(): tr[n][ff] = M
        meta[f] = sec; log(f'{arm} frame {f}: {sec:.0f}s')
    save(arm, tr, meta)


if __name__ == '__main__':
    arm = sys.argv[1]
    log('start', arm)
    if arm == 'S1': arm_s1()
    elif arm == 'S2': arm_s2()
    elif arm.startswith('S3'): arm_s3([int(x) for x in sys.argv[2].split(',')], tag=arm)   # e.g. S3a 25
    elif arm == 'P1': arm_p('P1', cofit=False)
    elif arm == 'P2': arm_p('P2')
    elif arm == 'P3': arm_p('P3', mu=0.01, d_min=0.0)
    elif arm == 'P3tol': arm_p('P3tol', mu=0.01, d_min=-2.0)
    elif arm == 'P2_drr': arm_p('P2_drr', w_f2f=0.0)
    elif arm == 'P2_f2f': arm_p('P2_f2f', w_drr=0.0)
    elif arm == 'PF': arm_pf('PF')
    else: raise SystemExit(f'unknown arm {arm}')
    log('done', arm)
