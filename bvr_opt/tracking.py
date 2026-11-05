"""Proposed method: keyframe-anchored gap fitting (and its per-frame variant).

Within a gap between keyframes a and b, each bone's pose is
    pose_n(f) = base_n(f) * Delta_n(f),   Delta_n(f) = sum_k c_nk sin(k pi (f - a) / (b - a)),  k = 1..K
where base_n is the interpolated keyframe path and Delta a 6-DOF pivot-centred increment (mm, deg) that
vanishes at both keyframes (hard anchors). The coefficients of all co-fitted bones are fitted jointly to

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


class GapFit:
    def __init__(self, trial, views, fovs, f2f, sdfs, names, a, b, base, cofit=True, K=3,
                 w_drr=1.0, w_f2f=2.0, lam=0.003, mu=0.0, d_min=0.0,
                 res=(256, 512), iters=(40, 50), lrs=(0.3, 0.1), batch=4, seed=0, log=print):
        """views: {res: [View per camera]}; base: {(n, f): 4x4} for f in a..b (a, b = keyframe poses)."""
        self.T, self.views, self.fovs, self.f2f, self.sdfs = trial, views, fovs, f2f, sdfs
        self.names, self.a, self.b, self.cofit, self.K = list(names), a, b, cofit, K
        self.w = dict(drr=w_drr, f2f=w_f2f, lam=lam, mu=mu); self.d_min = d_min
        self.sched = list(zip(res, iters, lrs)); self.batch = batch
        self.rng = np.random.default_rng(seed); self.log = log
        self.frames = list(range(a, b + 1)); self.inner = self.frames[1:-1]
        self.basis = {f: torch.tensor([np.sin(k * np.pi * (f - a) / (b - a)) for k in range(1, K + 1)],
                                      dtype=torch.float64) for f in self.frames}
        self.base = {(n, f): Pose.from_matrix(base[(n, f)], trial.bones[n].pivot) for n in names for f in self.frames}
        self.C = {n: torch.zeros(K, 6, dtype=torch.float64, requires_grad=True) for n in names}
        self.pts = {n: torch.tensor(trial.bones[n].surface_points(2500, 0.6, seed=0)) for n in names}

    def pose(self, n, f):
        if f in (self.a, self.b):
            return self.base[(n, f)].R, self.base[(n, f)].t
        return self.base[(n, f)].apply(self.basis[f] @ self.C[n])

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
        return sum(collision.penalty(self.sdfs, {n: self.pose(n, f) for n in self.names}, self.d_min)
                   for f in self.inner) / len(self.inner)

    def step(self, res, fs):
        """Accumulate gradients of the full objective term by term (one frame's render graph at a time,
        to bound memory). Returns the loss value and its terms."""
        terms = {}
        L = self.w['lam'] * sum((c ** 2).sum() for c in self.C.values()); L.backward(); total = float(L.detach())
        if self.w['drr'] > 0:
            d = 0.0
            for f in fs:
                t_ = self.w['drr'] * self.drr_frame(res, f) / len(fs)
                if torch.is_tensor(t_):
                    t_.backward(); d += float(t_)
            terms['drr'] = d / self.w['drr']; total += d
        if self.w['f2f'] > 0:
            t_ = self.f2f_term(); (self.w['f2f'] * t_).backward()
            terms['f2f'] = float(t_); total += self.w['f2f'] * float(t_)
        if self.w['mu'] > 0 and len(self.names) > 1:
            t_ = self.contact_term(); (self.w['mu'] * t_).backward()
            terms['contact'] = float(t_); total += self.w['mu'] * float(t_)
        return total, terms

    def run(self):
        t0 = time.time(); hist = []
        for res, it, lr in self.sched:
            opt = torch.optim.Adam(list(self.C.values()), lr=lr)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it)
            for k in range(it):
                fs = self.rng.choice(self.inner, size=min(self.batch, len(self.inner)), replace=False)
                opt.zero_grad()
                L, terms = self.step(res, fs)
                opt.step(); sch.step()
                hist.append({'res': res, 'L': L, **terms})
            self.log(f'    gap {self.a}-{self.b} {"+".join(self.names)}: res {res} done, loss {L:.4f}, '
                     f'{time.time() - t0:.0f}s')
        track = {}
        with torch.no_grad():
            for n in self.names:
                for f in self.inner:
                    R, t = self.pose(n, f)
                    M = np.eye(4); M[:3, :3] = R.numpy(); M[:3, 3] = t.numpy(); track[(n, f)] = M
        return track, dict(sec=time.time() - t0, hist=hist, coef={n: c.detach().numpy().tolist() for n, c in self.C.items()})


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
            R, t = P0[n].apply(X[n]); M = np.eye(4); M[:3, :3] = R.numpy(); M[:3, 3] = t.numpy(); out[(n, f)] = M
    return out, time.time() - t0
