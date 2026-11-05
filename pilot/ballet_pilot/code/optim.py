"""Optimisers on the 6-vector increment [dx,dy,dz (mm), rotation vector (deg)] about the bone pivot."""
import numpy as np, torch, time
from scipy.optimize import minimize
from costs import frame_cost, make_views


def autoscoper_pso_nm(f, lim=3.0, max_epochs=100, max_stall=10, seed=0, offsets=0.1, verbose=False):
    """Faithful port of Autoscoper's PSO (PSO.cpp) followed by Nelder-Mead (DownhillSimplex.cpp).
    f: numpy(6) -> float. Returns (delta, cost, n_evals)."""
    rng = np.random.default_rng(seed)
    N, D, c1, c2, w = 100, 6, 1.5, 1.5, 0.8
    X = np.vstack([np.zeros(D), rng.uniform(-lim, lim, (N - 1, D))])
    V = np.zeros((N, D)); pb = X.copy(); pbc = np.full(N, np.inf)
    gb, gbc = np.zeros(D), np.inf
    evals, stall, epoch = 0, 0, 0
    while True:
        prev = gbc
        for i in range(N):
            rp, rg = rng.random(D), rng.random(D)
            V[i] = w * V[i] + c1 * rp * (pb[i] - X[i]) + c2 * rg * (gb - X[i])
            X[i] += V[i]
            c = f(X[i]); evals += 1
            if c < pbc[i]: pbc[i], pb[i] = c, X[i].copy()
            if c < gbc: gbc, gb = c, X[i].copy()
        w *= 0.9; epoch += 1
        d = abs(gbc - prev) if np.isfinite(prev) else np.inf
        if d < 1e-4: stall += 1
        elif d > 1e-3: stall = 0
        if verbose: print(f'  PSO epoch {epoch}: best {gbc:.4f}', flush=True)
        if stall >= max_stall or epoch > max_epochs: break
    # Nelder-Mead around PSO result (Autoscoper re-centres on it); simplex offsets 0.1
    sim = np.vstack([gb] + [gb + offsets * np.eye(D)[k] for k in range(D)])
    cnt = [0]
    def g(x): cnt[0] += 1; return f(x)
    r = minimize(g, gb, method='Nelder-Mead', options=dict(initial_simplex=sim, xatol=1e-3, fatol=1e-7, maxiter=1000))
    best = r.x if r.fun < gbc else gb
    return best, min(r.fun, gbc), evals + cnt[0]


def gradient_refine(cost_at_res, resolutions=(256, 512, 512), iters=(40, 40, 40), lr=(0.3, 0.15, 0.05), x0=None):
    """Adam on the differentiable cost, coarse-to-fine render resolution.
    cost_at_res(res, delta_tensor) -> scalar tensor. Returns (delta, cost, n_evals)."""
    x = torch.zeros(6, dtype=torch.float64) if x0 is None else torch.tensor(x0, dtype=torch.float64)
    x.requires_grad_(True)
    n = 0; best = None
    for res, it, a in zip(resolutions, iters, lr):
        opt = torch.optim.Adam([x], lr=a)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it)
        for k in range(it):
            opt.zero_grad()
            c = cost_at_res(res, x); c.backward(); n += 1
            if res == resolutions[-1] and (best is None or c.item() < best[1]):
                best = (x.detach().clone().numpy(), c.item())
            opt.step(); sched.step()
    with torch.no_grad():
        c = cost_at_res(resolutions[-1], x).item()
    if c < best[1]:
        best = (x.detach().numpy().copy(), c)
    return best[0], best[1], n
