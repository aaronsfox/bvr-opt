"""Proposed method: keyframe-anchored gap fitting (and its per-frame variant).

Within a gap between keyframes a and b, each bone's pose is
    pose_n(f) = base_n(f) * Delta_n(f),   Delta_n(f) = sum_k c_nk sin(k pi (f - a) / (b - a)),  k = 1..K
where base_n is the interpolated keyframe path and Delta a 6-DOF pivot-centred increment (mm, deg) that
vanishes at both keyframes (hard anchors). With a keyframe tolerance, each keyframe also gets its own
correction, blended linearly across the gaps either side (see AnchoredFit). The coefficients of all
co-fitted bones are fitted jointly to

    L = w_drr * DRR term + w_f2f * frame-to-frame term + lam * sum |c|^2 + mu * contact penalty

DRR term: per frame and camera, every co-fitted bone is rendered once over the union ROI and the parts are
summed into one DRR (overlap modelled). Each bone contributes 1 - signed gradient correlation between that
joint DRR and the radiograph inside its own dilated silhouette x the field-of-view mask. With cofit=False
a bone is rendered alone (current single-bone practice).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import collision
from .costs import gradcorr, sample_mask
from .filters import blur
from .device import to_np
from .geometry import Pose


def drr_term(views, rads, fovs, bones, poses, dil=3):
    """bones: {n: Bone}; poses: {n: (R, t)} (differentiable). Returns sum over bones and cameras of
    1 - gradient correlation, and the number of (bone, camera) terms."""
    tot, cnt = 0, 0
    names = list(poses)
    for v, rad, fov in zip(views, rads, fovs):
        rois = {n: v.roi(bones[n], *poses[n]) for n in names}
        rois = {n: r for n, r in rois.items() if (r[1] - r[0]).min() >= 3 * v.px}
        if not rois:
            continue
        lo, hi = v.union_roi(list(rois.values()))
        rays = v._rays(lo, hi)
        parts = v.render_parts([(bones[n], *poses[n]) for n in rois], lo, hi, rays)
        joint = blur(torch.clamp(sum(parts), 0, 1), 1.0)
        r = blur(1 - v.sample_image(rad, lo, hi).double(), 1.0)
        fm = sample_mask(v, fov, lo, hi).double()
        for n, p in zip(rois, parts):
            m = F.max_pool2d((p.detach() > 0).float()[None, None], 2 * dil + 1, 1, dil)[0, 0].double() * fm
            if m.sum() < 20:
                continue
            tot = tot + 1 - gradcorr(joint, r, m); cnt += 1
    return tot, cnt


def _capped(u, cap):
    """Smooth norm cap: |result| < cap, ~u for |u| << 1 (u is in units of cap)."""
    return cap * u / torch.sqrt(1 + (u * u).sum())


class AnchoredFit:
    """Keyframe-anchored fit over one or more consecutive gaps (keyframes k_0 < k_1 < ... < k_m).

    pose_n(f) = base_n(f) * Delta_n(f), for f in gap (a, b) with s = (f - a) / (b - a):
        Delta_n(f) = (1 - s) delta_n(a) + s delta_n(b) + sum_k c_nk(gap) sin(k pi s)
    delta_n(k) is the keyframe tolerance: a 6-DOF correction of keyframe k (shared by the gaps either side).
    key_tol = None (or 0) -> hard anchors, delta = 0 and keyframes are not fitted; gaps are then independent.
    key_tol = (mm, deg) -> each keyframe may move within a ball of key_tol[0] mm (translation of the pivot)
    and key_tol[1] deg (rotation), via a smooth norm cap, plus a quadratic prior key_w * |delta / tol|^2.
    With tolerance, keyframe frames are also used in the DRR term (their images inform their pose).
    """

    def __init__(self, trial, views, fovs, f2f, sdfs, names, keys, base, cofit=True, K=3,
                 w_drr=1.0, w_f2f=2.0, lam=0.003, mu=0.0, d_min=0.0, key_tol=None, key_w=0.01,
                 res=(256, 512), iters=(40, 50), lrs=(0.3, 0.1), batch=4, seed=0, log=print):
        """views: {res: [View per camera]}; base: {(n, f): 4x4} for every f in keys[0]..keys[-1]
        (keyframe entries are the keyframe poses as given, e.g. by a rater)."""
        self.T, self.views, self.fovs, self.f2f, self.sdfs = trial, views, fovs, f2f, sdfs
        self.names, self.keys, self.cofit, self.K = list(names), list(keys), cofit, K
        self.gaps = list(zip(self.keys[:-1], self.keys[1:]))
        self.w = dict(drr=w_drr, f2f=w_f2f, lam=lam, mu=mu, key=key_w); self.d_min = d_min
        self.tol = None if not key_tol or max(key_tol) <= 0 else (float(key_tol[0]), float(key_tol[1]))
        self.sched = list(zip(res, iters, lrs)); self.batch = batch
        self.rng = np.random.default_rng(seed); self.log = log
        self.frames = list(range(self.keys[0], self.keys[-1] + 1))
        self.gap_of = {}
        for g, (a, b) in enumerate(self.gaps):
            for f in range(a + 1, b):
                self.gap_of[f] = g
        self.inner = [f for f in self.frames if f in self.gap_of]
        self.fit_frames = self.frames if self.tol else self.inner            # frames whose pose is fitted
        self.base = {(n, f): Pose.from_matrix(base[(n, f)], trial.bones[n].pivot) for n in names for f in self.frames}
        self.C = {(n, g): torch.zeros(K, 6, dtype=torch.float64, requires_grad=True)
                  for n in names for g in range(len(self.gaps))}
        self.U = {(n, k): torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
                  for n in names for k in self.keys} if self.tol else {}
        self.pts = {n: torch.tensor(trial.bones[n].surface_points(2500, 0.6, seed=0)) for n in names}

    def params(self):
        return list(self.C.values()) + list(self.U.values())

    def key_delta(self, n, k):
        if not self.tol:
            return None
        u = self.U[(n, k)]
        return torch.cat([_capped(u[0], self.tol[0]), _capped(u[1], self.tol[1])])

    def delta(self, n, f):
        if f in self.gap_of:
            g = self.gap_of[f]; a, b = self.gaps[g]; s = (f - a) / (b - a)
            basis = torch.tensor([np.sin(k * np.pi * s) for k in range(1, self.K + 1)], dtype=torch.float64)
            d = basis @ self.C[(n, g)]
            if self.tol:
                d = d + (1 - s) * self.key_delta(n, a) + s * self.key_delta(n, b)
            return d
        return self.key_delta(n, f)                                        # keyframe (None: hard anchor)

    def pose(self, n, f):
        d = self.delta(n, f)
        if d is None:
            return self.base[(n, f)].R, self.base[(n, f)].t
        return self.base[(n, f)].apply(d)

    def drr_frame(self, res, f):
        """Mean DRR term over the (bone, camera) pairs of frame f."""
        rads = [self.T.image(ci, f, res) for ci in range(len(self.T.cams))]
        poses = {n: self.pose(n, f) for n in self.names}
        groups = [self.names] if self.cofit else [[n] for n in self.names]
        tot, cnt = 0, 0
        for g in groups:
            t_, c_ = drr_term(self.views[res], rads, self.fovs, self.T.bones, {n: poses[n] for n in g})
            tot, cnt = tot + t_, cnt + c_
        return tot / max(cnt, 1)

    def f2f_term(self):
        tot, cnt = 0, 0
        for n in self.names:
            W = {}
            for f in self.frames:
                R, t = self.pose(n, f); W[f] = self.pts[n] @ R.T + t
            for f in self.frames[:-1]:
                t_, c_ = self.f2f.pair(W[f], W[f + 1], f)
                tot, cnt = tot + t_, cnt + c_
        return tot / max(cnt, 1)

    def contact_term(self):
        fs = self.fit_frames
        return sum(collision.penalty(self.sdfs, {n: self.pose(n, f) for n in self.names}, self.d_min)
                   for f in fs) / len(fs)

    def prior(self):
        L = self.w['lam'] * sum((c ** 2).sum() for c in self.C.values())
        if self.tol:
            L = L + self.w['key'] * sum(((self.key_delta(n, k)[:3] / self.tol[0]) ** 2).sum() +
                                        ((self.key_delta(n, k)[3:] / self.tol[1]) ** 2).sum()
                                        for n, k in self.U)
        return L

    def sample_frames(self):
        """`batch` frames per gap (inner frames; with tolerance, the gap's end keyframes as well)."""
        fs = set()
        for a, b in self.gaps:
            pool = list(range(a, b + 1)) if self.tol else list(range(a + 1, b))
            fs.update(int(x) for x in self.rng.choice(pool, size=min(self.batch, len(pool)), replace=False))
        return sorted(fs)

    def step(self, res, fs):
        """Accumulate gradients of the objective term by term (one frame's render graph at a time, to bound
        memory). DRR term: mean over the sampled frames, scaled by the number of gaps so that each gap's
        coefficients see the same gradient scale as a single-gap fit. Returns the loss value and its terms."""
        terms = {}
        L = self.prior(); L.backward(); total = float(L.detach())
        ng = len(self.gaps)
        if self.w['drr'] > 0:
            d = 0.0
            for f in fs:
                t_ = self.w['drr'] * ng * self.drr_frame(res, f) / len(fs)
                if torch.is_tensor(t_):
                    t_.backward(); d += float(t_.detach())
            terms['drr'] = d / self.w['drr'] / ng; total += d
        if self.w['f2f'] > 0:
            t_ = self.f2f_term(); (self.w['f2f'] * ng * t_).backward()
            t_ = float(t_.detach()); terms['f2f'] = t_; total += self.w['f2f'] * ng * t_
        if self.w['mu'] > 0 and len(self.names) > 1:
            t_ = self.contact_term(); (self.w['mu'] * ng * t_).backward()
            t_ = float(t_.detach()); terms['contact'] = t_; total += self.w['mu'] * ng * t_
        return total, terms

    def run(self):
        t0 = time.time(); hist = []
        tag = f'keys {self.keys[0]}-{self.keys[-1]} ({len(self.gaps)} gaps) {"+".join(self.names)}'
        for res, it, lr in self.sched:
            opt = torch.optim.Adam(self.params(), lr=lr)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it)
            for k in range(it):
                fs = self.sample_frames()
                opt.zero_grad()
                L, terms = self.step(res, fs)
                opt.step(); sch.step()
                hist.append({'res': res, 'L': L, **terms})
            self.log(f'    {tag}: res {res} done, loss {L:.4f}, {time.time() - t0:.0f}s')
        track = {}
        with torch.no_grad():
            for n in self.names:
                for f in self.fit_frames:
                    R, t = self.pose(n, f)
                    M = np.eye(4); M[:3, :3] = to_np(R); M[:3, 3] = to_np(t); track[(n, f)] = M
        info = dict(sec=time.time() - t0, hist=hist,
                    coef={f'{n}_{g}': to_np(c).tolist() for (n, g), c in self.C.items()})
        if self.tol:
            with torch.no_grad():
                info['key_delta'] = {f'{n}_{k}': to_np(self.key_delta(n, k)).tolist() for n, k in self.U}
        return track, info


def GapFit(trial, views, fovs, f2f, sdfs, names, a, b, base, **kw):
    """Single gap with hard anchors (pilot interface)."""
    return AnchoredFit(trial, views, fovs, f2f, sdfs, names, [a, b], base, **kw)


def per_frame_fit(trial, views, fovs, sdfs, names, f, base, cofit=True, mu=0.0, d_min=0.0,
                  res=(256, 512), iters=(40, 50), lrs=(0.3, 0.1)):
    """Unanchored ablation: each frame fitted on its own (DRR term only, plus contact if mu > 0),
    starting from the interpolated pose base[(n, f)]."""
    P0 = {n: Pose.from_matrix(base[(n, f)], trial.bones[n].pivot) for n in names}
    X = {n: torch.zeros(6, dtype=torch.float64, requires_grad=True) for n in names}
    t0 = time.time()
    for r, it, lr in zip(res, iters, lrs):
        opt = torch.optim.Adam(list(X.values()), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it)
        rads = [trial.image(ci, f, r) for ci in range(len(trial.cams))]
        for k in range(it):
            opt.zero_grad()
            poses = {n: P0[n].apply(X[n]) for n in names}
            groups = [names] if cofit else [[n] for n in names]
            tot, cnt = 0, 0
            for g in groups:
                t_, c_ = drr_term(views[r], rads, fovs, trial.bones, {n: poses[n] for n in g})
                tot, cnt = tot + t_, cnt + c_
            L = tot / max(cnt, 1)
            if mu > 0 and len(names) > 1:
                L = L + mu * collision.penalty(sdfs, poses, d_min)
            L.backward(); opt.step(); sch.step()
    out = {}
    with torch.no_grad():
        for n in names:
            R, t = P0[n].apply(X[n]); M = np.eye(4); M[:3, :3] = to_np(R); M[:3, 3] = to_np(t); out[(n, f)] = M
    return out, time.time() - t0


# ------------------------------------------------------------------------------------------- stepwise chains
def step_fit(trial, views, fovs, f2f, sdfs, names, f, init, pts, prev=None, prev_frame=None, w_drr=1.0, w_f2f=2.0,
             mu=0.0, d_min=0.0, res=(256, 512), iters=(20, 20), lrs=(0.2, 0.05)):
    """Fit frame f on its own (bones co-fitted), starting from init {n: 4x4}. If prev {n: 4x4} is given (the
    already-fitted neighbouring frame prev_frame = f - 1 or f + 1, held fixed), the frame-to-frame term links
    f to it. Objective: w_drr * DRR term + w_f2f * frame-to-frame term + mu * contact. Returns {n: 4x4}."""
    P0 = {n: Pose.from_matrix(init[n], trial.bones[n].pivot) for n in names}
    X = {n: torch.zeros(6, dtype=torch.float64, requires_grad=True) for n in names}
    prevW = {}
    if prev is not None and w_f2f > 0:
        for n in names:
            M = torch.tensor(prev[n], dtype=torch.float64); prevW[n] = pts[n] @ M[:3, :3].T + M[:3, 3]
    for r, it, lr in zip(res, iters, lrs):
        opt = torch.optim.Adam(list(X.values()), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it)
        rads = [trial.image(ci, f, r) for ci in range(len(trial.cams))]
        for k in range(it):
            opt.zero_grad()
            poses = {n: P0[n].apply(X[n]) for n in names}
            L = 0
            if w_drr > 0:
                t_, c_ = drr_term(views[r], rads, fovs, trial.bones, poses)
                L = L + w_drr * t_ / max(c_, 1)
            if prevW:
                tot, cnt = 0, 0
                for n in names:
                    R, t = poses[n]; W = pts[n] @ R.T + t
                    t_, c_ = f2f.pair(prevW[n], W, prev_frame) if prev_frame < f else f2f.pair(W, prevW[n], f)
                    tot, cnt = tot + t_, cnt + c_
                L = L + w_f2f * tot / max(cnt, 1)
            if mu > 0 and len(names) > 1:
                L = L + mu * collision.penalty(sdfs, poses, d_min)
            if torch.is_tensor(L) and L.requires_grad:
                L.backward(); opt.step()
            sch.step()
    out = {}
    with torch.no_grad():
        for n in names:
            R, t = P0[n].apply(X[n]); M = np.eye(4); M[:3, :3] = to_np(R); M[:3, 3] = to_np(t); out[n] = M
    return out


def chain(trial, views, fovs, f2f, sdfs, names, key, frames, direction, first_guess, pts, **kw):
    """Step frame by frame from keyframe `key` through `frames` (in stepping order, e.g. a+1, a+2, ... or
    b-1, b-2, ...). Each frame starts from a constant-velocity prediction from the two previous poses
    (the first frame from first_guess {n: 4x4}, e.g. the spline pose) and is fitted with step_fit, linked
    to the previous frame by the frame-to-frame term. key: {n: 4x4} at the keyframe; direction: +1 stepping
    forward from it, -1 stepping backward. Returns {(n, f): 4x4}."""
    hist = [(frames[0] - direction, key)]     # (frame, poses): the keyframe comes just before the first frame
    out = {}
    for i, f in enumerate(frames):
        if i == 0:
            init = first_guess
        else:
            M1, M2 = hist[-1][1], hist[-2][1]                     # constant velocity: M1 * (M2^-1 * M1)
            init = {n: M1[n] @ np.linalg.inv(M2[n]) @ M1[n] for n in names}
        fit = step_fit(trial, views, fovs, f2f, sdfs, names, f, init, pts, prev=hist[-1][1], prev_frame=hist[-1][0], **kw)
        hist.append((f, fit))
        for n in names: out[(n, f)] = fit[n]
    return out


def blend_poses(Mf, Mb, s, pivot):
    """Pose between Mf (weight 1 - s) and Mb (weight s): pivot position linear, rotation geodesic."""
    from scipy.spatial.transform import Rotation as Rot
    Rf, Rb = Mf[:3, :3], Mb[:3, :3]
    R = Rf @ Rot.from_rotvec(s * Rot.from_matrix(Rf.T @ Rb).as_rotvec()).as_matrix()
    p = (1 - s) * (Rf @ pivot + Mf[:3, 3]) + s * (Rb @ pivot + Mb[:3, 3])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p - R @ pivot
    return M


# ------------------------------------------------------------------------------------------- keyframe correction
class KeyframeFit:
    """Correct one rater keyframe k using the images of frames k - window .. k + window.

    Each bone's pose over the window is pose(f) = base(f) * Delta(f), with s = (f - k) / window and
        Delta(f) = delta + s * v + s^2 * w
    where delta is the correction of the keyframe itself, kept inside a ball of tol = (mm, deg) by a smooth
    norm cap (plus a weak quadratic prior), and v, w (unbounded) let the neighbouring frames follow their
    own path. Bones are co-fitted; objective = DRR term over the window + frame-to-frame term between
    consecutive window frames + contact + priors. Only the corrected keyframe pose is kept.
    base: {(n, f): 4x4} for every window frame (e.g. the spline through the rater's keyframes)."""

    def __init__(self, trial, views, fovs, f2f, sdfs, names, k, frames, base, tol=(1.0, 2.0), w_drr=1.0,
                 w_f2f=2.0, mu=0.0, d_min=0.0, key_w=0.01, lam=0.003, res=(256, 512), iters=(40, 50),
                 lrs=(0.3, 0.1), log=print):
        self.T, self.views, self.fovs, self.f2f, self.sdfs = trial, views, fovs, f2f, sdfs
        self.names, self.k, self.frames = list(names), k, list(frames)
        self.half = max(max(abs(f - k) for f in self.frames), 1)
        self.tol = (float(tol[0]), float(tol[1]))
        self.w = dict(drr=w_drr, f2f=w_f2f, mu=mu, key=key_w, lam=lam); self.d_min = d_min
        self.sched = list(zip(res, iters, lrs)); self.log = log
        self.base = {(n, f): Pose.from_matrix(base[(n, f)], trial.bones[n].pivot) for n in names for f in self.frames}
        self.U = {n: torch.zeros(2, 3, dtype=torch.float64, requires_grad=True) for n in names}
        self.V = {n: torch.zeros(2, 6, dtype=torch.float64, requires_grad=True) for n in names}
        self.pts = {n: torch.tensor(trial.bones[n].surface_points(2500, 0.6, seed=0)) for n in names}

    def params(self):
        return list(self.U.values()) + list(self.V.values())

    def delta_key(self, n):
        u = self.U[n]
        return torch.cat([_capped(u[0], self.tol[0]), _capped(u[1], self.tol[1])])

    def pose(self, n, f):
        s = (f - self.k) / self.half
        d = self.delta_key(n) + s * self.V[n][0] + s * s * self.V[n][1]
        return self.base[(n, f)].apply(d)

    def step(self, res):
        L = self.w['lam'] * sum((v ** 2).sum() for v in self.V.values())
        L = L + self.w['key'] * sum(((self.delta_key(n)[:3] / self.tol[0]) ** 2).sum() +
                                    ((self.delta_key(n)[3:] / self.tol[1]) ** 2).sum() for n in self.names)
        L.backward(); total = float(L.detach())
        nf = len(self.frames)
        if self.w['drr'] > 0:                                 # one frame's render graph at a time (memory)
            for f in self.frames:
                rads = [self.T.image(ci, f, res) for ci in range(len(self.T.cams))]
                t_, c_ = drr_term(self.views[res], rads, self.fovs, self.T.bones, {n: self.pose(n, f) for n in self.names})
                if torch.is_tensor(t_):
                    t_ = self.w['drr'] * t_ / max(c_, 1) / nf; t_.backward(); total += float(t_.detach())
        if self.w['f2f'] > 0 and nf > 1:
            tot, cnt = 0, 0
            for n in self.names:
                W = {f: self.pts[n] @ self.pose(n, f)[0].T + self.pose(n, f)[1] for f in self.frames}
                for f in self.frames[:-1]:
                    t_, c_ = self.f2f.pair(W[f], W[f + 1], f); tot, cnt = tot + t_, cnt + c_
            if torch.is_tensor(tot):
                t_ = self.w['f2f'] * tot / max(cnt, 1); t_.backward(); total += float(t_.detach())
        if self.w['mu'] > 0 and len(self.names) > 1:
            t_ = self.w['mu'] * sum(collision.penalty(self.sdfs, {n: self.pose(n, f) for n in self.names}, self.d_min)
                                    for f in self.frames) / nf
            t_.backward(); total += float(t_.detach())
        return total

    def run(self):
        t0 = time.time()
        for res, it, lr in self.sched:
            opt = torch.optim.Adam(self.params(), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it)
            for _ in range(it):
                opt.zero_grad(); L = self.step(res); opt.step(); sch.step()
        out = {}
        with torch.no_grad():
            for n in self.names:
                R, t = self.pose(n, self.k); M = np.eye(4); M[:3, :3] = to_np(R); M[:3, 3] = to_np(t); out[n] = M
            moved = {n: to_np(self.delta_key(n)).tolist() for n in self.names}
        self.log(f'    keyframe {self.k} (frames {self.frames[0]}-{self.frames[-1]}): loss {L:.4f}, {time.time() - t0:.0f}s')
        return out, dict(delta=moved, sec=time.time() - t0)
