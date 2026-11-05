"""Replica of Autoscoper's optimisers on the 6-vector pose increment [dx, dy, dz (mm), rotation vector (deg)]
about the bone pivot (libautoscoper/src/PSO.cpp, DownhillSimplex.cpp, Tracker::optimize; Autoscoper main,
Nov 2026). Defaults are those of Autoscoper's tracking dialog (TrackingOptionsDialog.cpp):
PSO limits -3..3, max epochs 1000, max stall 25; simplex offsets from the cfg (OptimizationOffsets 0.1).
"""
import numpy as np

NUM_PARTICLES, C1, C2, D = 100, 1.5, 1.5, 6


def pso(f, lo=-3.0, hi=3.0, max_epochs=1000, max_stall=25, seed=0, log=None):
    """Autoscoper PSO. Particle 0 starts at the initial pose (zero increment), the rest uniformly in
    [lo, hi]^6; velocities start at zero; inertia 0.8, multiplied by 0.9 per epoch; personal and global
    bests updated particle by particle (asynchronous). Stops after `max_stall` epochs whose global-best
    change is < 1e-4 (a change > 1e-3 resets the count) or after max_epochs + 1 epochs."""
    rng = np.random.default_rng(seed)
    X = np.vstack([np.zeros(D), lo + rng.random((NUM_PARTICLES - 1, D)) * (hi - lo)])
    V = np.zeros_like(X)
    pb, pbc = X.copy(), np.full(NUM_PARTICLES, np.inf)
    gb, gbc = np.zeros(D), np.inf          # Particle() default: position 0, NCC = FLT_MAX
    omega, stall, epoch, evals = 0.8, 0, 0, 0
    while True:
        last = epoch >= max_epochs
        prev = gbc
        for i in range(NUM_PARTICLES):
            rp, rg = rng.random(D), rng.random(D)
            V[i] = omega * V[i] + C1 * rp * (pb[i] - X[i]) + C2 * rg * (gb - X[i])
            X[i] = X[i] + V[i]
            c = f(X[i]); evals += 1
            if c < pbc[i]: pbc[i], pb[i] = c, X[i].copy()
            if c < gbc: gbc, gb = c, X[i].copy()
        omega *= 0.9
        ch = abs(gbc - prev) if np.isfinite(prev) else np.inf
        if ch < 1e-4: stall += 1
        elif ch > 1e-3: stall = 0
        epoch += 1
        if log: log(f'    PSO epoch {epoch}: best {gbc:.5f}')
        if stall == max_stall or last:
            break
    return gb, gbc, evals, epoch


def amoeba(f, x0, offsets=0.1, ftol=1e-3, itmax=1000):
    """Autoscoper's AMOEBA (Numerical Recipes Nelder-Mead). Note Autoscoper's exit test is on the simplex
    size in parameter space: mean distance of the vertices to their centroid < ftol."""
    x0 = np.asarray(x0, float)
    off = np.broadcast_to(np.asarray(offsets, float), (D,))
    P = np.vstack([x0] + [x0 + np.eye(D)[k] * off[k] for k in range(D)])
    Y = np.array([f(p) for p in P]); evals = D + 1
    ALPHA, BETA, GAMMA = 1.0, 0.5, 2.0
    it = 0
    while True:
        ilo = int(np.argmin(Y)); order = np.argsort(Y); ihi = int(order[-1]); inhi = int(order[-2])
        cent = P.mean(0)
        rtol = np.mean(np.linalg.norm(P - cent, axis=1))
        if rtol < ftol or it == itmax:
            break
        it += 1
        pbar = (P.sum(0) - P[ihi]) / D
        pr = (1 + ALPHA) * pbar - ALPHA * P[ihi]; ypr = f(pr); evals += 1
        if ypr <= Y[ilo]:
            prr = GAMMA * pr + (1 - GAMMA) * pbar; yprr = f(prr); evals += 1
            if yprr < Y[ilo]: P[ihi], Y[ihi] = prr, yprr
            else: P[ihi], Y[ihi] = pr, ypr
        elif ypr >= Y[inhi]:
            if ypr < Y[ihi]: P[ihi], Y[ihi] = pr, ypr
            prr = BETA * P[ihi] + (1 - BETA) * pbar; yprr = f(prr); evals += 1
            if yprr < Y[ihi]: P[ihi], Y[ihi] = prr, yprr
            else:
                for i in range(D + 1):
                    if i != ilo:
                        P[i] = 0.5 * (P[i] + P[ilo]); Y[i] = f(P[i]); evals += 1
        else:
            P[ihi], Y[ihi] = pr, ypr
    return P[0], Y[0], evals, it      # Autoscoper takes vertex 1 (P[1]+1), not the best vertex


def autoscoper_optimize(make_f, method='pso', seed=0, offsets=0.1, log=None, **pso_kw):
    """One Autoscoper 'optimize frame'.
    make_f(base_delta) -> f(x) evaluating the cost at pose  current * V^-1 * (base_delta) * V * V^-1 * x * V,
    i.e. x is an increment on top of an already-applied increment (composition handled by the caller).

    method='pso':     PSO from the current pose, pose moved to the PSO best, then simplex from there;
                      the simplex result is kept only if it beats the cost at the *initial* pose
                      (Tracker::optimize), otherwise the PSO pose is kept.
    method='simplex': simplex only.
    Returns dict(stages=[increment vectors to compose in order], cost, evals)."""
    f0 = make_f(None)
    init = f0(np.zeros(D))
    if method == 'simplex':
        x, c, ne, it = amoeba(f0, np.zeros(D), offsets)
        return dict(stages=[x], cost=float(c), init=float(init), evals=ne + 1, iters=it)
    g, gc, ne1, ep = pso(f0, seed=seed, log=log, **pso_kw)
    f1 = make_f(g)
    x, c, ne2, it = amoeba(f1, np.zeros(D), offsets)
    final = f1(x)
    stages = [g, x] if final < init else [g]
    return dict(stages=stages, cost=float(min(final, gc) if final < init else gc), init=float(init),
                pso_cost=float(gc), evals=ne1 + ne2 + 2, epochs=ep, iters=it)
