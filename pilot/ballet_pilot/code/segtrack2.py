"""Keyframe-anchored segment tracking with a model-based frame-to-frame (F2F) term.

pose_n(f) = interp_n(f) * Delta_n(f),  Delta_n(f) = sum_k c_nk sin(k pi (f-a)/(b-a))   (zero at both keyframes)

Loss = w_drr * mean_f [ DRR term: 1 - gradient correlation(DRR, radiograph), neighbouring bones rendered, FOV mask ]
     + w_f2f * mean_{f,f+1,cam} [ 1 - NCC of signed image gradients sampled at bone points projected with pose(f) in
                                  image f and with pose(f+1) in image f+1 ]
     + lam * |c|^2
The F2F term is a Lucas-Kanade-style photometric consistency with a 3D rigid warp shared by both cameras.
Usage: python3 segtrack2.py <w_drr> <w_f2f> <a> <b> <bone1,bone2,...> <eval frames> [K]"""
exec(open('/root/work/holdout.py').read().split("kind,vn=")[0])
import json, costs
from variants import COMBOS, _blur, _sobel_xy
costs.USE_FOV = True
w_drr, w_f2f = float(sys.argv[1]), float(sys.argv[2])
a, b = int(sys.argv[3]), int(sys.argv[4]); names = sys.argv[5].split(',')
evalf = [int(x) for x in sys.argv[6].split(',')]
K = int(sys.argv[7]) if len(sys.argv) > 7 else 3
LAM, ITERS, BATCH, NPTS, SIG = 0.003, (40, 50), 4, 2500, 2.0
pre, sim = COMBOS['gradcorr']
inner = list(range(a + 1, b)); allf = list(range(a, b + 1))
basis = {f: torch.tensor([np.sin(k * np.pi * (f - a) / (b - a)) for k in range(1, K + 1)], dtype=torch.float64) for f in allf}
base = {(n, f): (pose(n, f) if f in (a, b) else interp(n, f, a, b)) for n in names for f in allf}
C = {n: torch.zeros(K, 6, dtype=torch.float64, requires_grad=True) for n in names}
def delta(n, f): return basis[f] @ C[n]

# ---- bone points for F2F: voxels on strong 3D gradients (cortex / edges) ----
rng = np.random.default_rng(0)
PTS = {}
for n in names:
    bn = bones[n]; v = bn.vol[0, 0].numpy()
    gz, gy, gx = np.gradient(v, bn.sz, bn.sy, bn.sx); g = np.sqrt(gx**2 + gy**2 + gz**2)
    k, r, c = np.nonzero((v > bn.cutoff) & (g > np.quantile(g[v > bn.cutoff], 0.6)))
    sel = rng.choice(len(k), size=min(NPTS, len(k)), replace=False)
    P = np.c_[(c[sel] + .5) * bn.sx, (bn.H - r[sel] - .5) * bn.sy, -(k[sel] + .5) * bn.sz]
    PTS[n] = torch.tensor(P, dtype=torch.float64)

# ---- per-frame, per-camera signed gradient images of the (blurred, inverted) radiograph, full resolution ----
GIM, FOVT = {}, [torch.tensor(np.load(f'/root/work/fov_c{i+1}.npy').astype(np.float32)) for i in range(2)]
for f in allf:
    for ci in range(2):
        from scipy.ndimage import gaussian_filter, sobel as nd_sobel
        r_ = rad(ci, f + 1).numpy(); im = gaussian_filter((1 - r_).astype(np.float32), SIG * r_.shape[0] / cams[ci].W)   # no grad needed; numpy saves memory
        gx = nd_sobel(im, axis=1); gy = -nd_sobel(im, axis=0)
        GIM[(f, ci)] = torch.from_numpy(np.stack([gx, gy]))[None]   # 1,2,H,W
cams_t = [(torch.tensor(c.R), torch.tensor(c.t), c) for c in cams]

def project(ci, P):
    Rc, tc, c = cams_t[ci]
    pc = (P - tc) @ Rc.T
    col = c.cx - c.f * pc[:, 0] / pc[:, 2]; row = c.H - c.cy + c.f * pc[:, 1] / pc[:, 2]
    return col, row

def sample(f, ci, col, row):
    W = cams[ci].W; H = cams[ci].H
    g = torch.stack([2 * col / W - 1, 2 * row / H - 1], -1).float()[None, None]
    val = F.grid_sample(GIM[(f, ci)], g, mode='bilinear', align_corners=False)[0, :, 0]          # 2,N
    inside = F.grid_sample(FOVT[ci][None, None], g, mode='nearest', align_corners=False)[0, 0, 0] > 0.5
    return val, inside

def ncc1(x, y):
    x = x - x.mean(); y = y - y.mean()
    return (x * y).sum() / torch.sqrt((x * x).sum() * (y * y).sum() + 1e-12)

def f2f_loss():
    tot, cnt = 0, 0
    for n in names:
        world = {}
        for f in allf:
            R, t = base[(n, f)].apply(delta(n, f)) if f not in (a, b) else (base[(n, f)].R, base[(n, f)].t)
            world[f] = PTS[n] @ R.T + t
        for f in allf[:-1]:
            for ci in range(2):
                v0, i0 = sample(f, ci, *project(ci, world[f]))
                v1, i1 = sample(f + 1, ci, *project(ci, world[f + 1]))
                m = i0 & i1
                s = 0.5 * (ncc1(v0[0][m], v1[0][m]) + ncc1(v0[1][m], v1[1][m]))
                tot = tot + (1 - s); cnt += 1
    return tot / cnt

def drr_loss(res, fs):
    tot = 0
    for f in fs:
        rads = [rad(0, f + 1), rad(1, f + 1)]
        with torch.no_grad():
            cur = {n: Pose(*base[(n, f)].apply(delta(n, f)), bones[n].centre) for n in names}
        for n in names:
            others = [(bones[m], cur[m]) for m in names if m != n]
            others += [(bones[m], pose(m, f)) for m in ['1stMet', '2ndMet', '1stPhal', '2ndPhal']
                       if m not in names and not np.isnan(T[m][f, 0, 0])]
            tot = tot + frame_cost(VIEWS[res], rads, bones[n], base[(n, f)], delta(n, f), pre,
                                   others=others, joint=bool(others), sim=sim)
    return tot / (len(fs) * len(names))

t0 = time.time(); hist = []
for res, it, lr in zip((256, 512), ITERS, (0.3, 0.1)):
    opt = torch.optim.Adam(list(C.values()), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it)
    for k in range(it):
        opt.zero_grad()
        L = LAM * sum((C[n] ** 2).sum() for n in names)
        if w_drr > 0: L = L + w_drr * drr_loss(res, rng.choice(inner, size=min(BATCH, len(inner)), replace=False))
        if w_f2f > 0: L = L + w_f2f * f2f_loss()
        L.backward(); opt.step(); sch.step(); hist.append(L.item())
    print(f'  res {res} done, loss {L.item():.4f}, {time.time()-t0:.0f}s', flush=True)

tag = f'drr{w_drr:g}_f2f{w_f2f:g}_K{K}' + (f"_ds{__import__('os').environ['DS']}" if __import__('os').environ.get('DS') else '')
out = {'w_drr': w_drr, 'w_f2f': w_f2f, 'K': K, 'a': a, 'b': b, 'bones': names, 'sec': time.time() - t0, 'eval': [], 'track': {}, 'loss': hist}
for n in names:
    out['track'][n] = {}
    for f in inner:
        with torch.no_grad(): R, t = base[(n, f)].apply(delta(n, f))
        out['track'][n][f] = (R.numpy().tolist(), t.numpy().tolist())
        if f in evalf and not np.isnan(T[n][f, 0, 0]):
            P = Pose(R, t, bones[n].centre); e0 = err(n, base[(n, f)], f); e1 = err(n, P, f)
            out['eval'].append(dict(bone=n, frame=f, interp_mm=e0[0], interp_deg=e0[1], seg_mm=e1[0], seg_deg=e1[1]))
            print(f'f2f:{tag} {"+".join(names)} {a}-{b} {n} @{f}: interp {e0[0]:.2f}mm {e0[1]:.2f}deg -> {e1[0]:.2f}mm {e1[1]:.2f}deg', flush=True)
json.dump(out, open(f'f2f_{tag}_{"+".join(names)}_{a}_{b}.json', 'w'))
print('F2FRUNDONE', flush=True)
