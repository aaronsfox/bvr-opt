"""Forward-only tracking from a single start keyframe (no end anchor).
Sliding window of W frames after the last committed frame. Each window frame gets a free 6-vector increment about the
pivot, starting from a constant-velocity extrapolation. Loss over the window:
   w_drr * DRR gradient-correlation term (neighbouring bones rendered, FOV mask)
 + w_f2f * frame-to-frame signed-gradient NCC for consecutive pairs (last committed frame -> window)
 + w_acc * squared accelerations (bone-centre position, mm; rotation, deg)
Then the oldest window frame is committed and the window slides on.
Usage: python3 fwdtrack.py <start> <end> <bone1,bone2,...> [w_drr w_f2f w_acc W]"""
exec(open('/root/work/holdout.py').read().split("kind,vn=")[0])
import json, costs, os
from variants import COMBOS
from scipy.ndimage import gaussian_filter, sobel as nd_sobel
costs.USE_FOV = True
start, end = int(sys.argv[1]), int(sys.argv[2]); names = sys.argv[3].split(',')
w_drr, w_f2f, w_acc, W = (float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (1.0, 2.0, 0.02, 3)
ITERS, NPTS, SIG = (20, 10), 2500, 2.0
pre, sim = COMBOS['gradcorr']
rng = np.random.default_rng(0)

# bone points for the F2F term
PTS = {}
for n in names:
    bn = bones[n]; v = bn.vol[0, 0].numpy()
    gz, gy, gx = np.gradient(v, bn.sz, bn.sy, bn.sx); g = np.sqrt(gx**2 + gy**2 + gz**2)
    k, r, c = np.nonzero((v > bn.cutoff) & (g > np.quantile(g[v > bn.cutoff], 0.6)))
    sel = rng.choice(len(k), size=min(NPTS, len(k)), replace=False)
    PTS[n] = torch.tensor(np.c_[(c[sel] + .5) * bn.sx, (bn.H - r[sel] - .5) * bn.sy, -(k[sel] + .5) * bn.sz], dtype=torch.float64)
GIM = {}
FOVT = [torch.tensor(np.load(f'/root/work/fov_c{i+1}.npy').astype(np.float32)) for i in range(2)]
def gim(f, ci):
    if (f, ci) not in GIM:
        r_ = rad(ci, f + 1).numpy(); im = gaussian_filter((1 - r_).astype(np.float32), SIG * r_.shape[0] / cams[ci].W)
        GIM[(f, ci)] = torch.from_numpy(np.stack([nd_sobel(im, axis=1), -nd_sobel(im, axis=0)]))[None]
    return GIM[(f, ci)]
cams_t = [(torch.tensor(c.R), torch.tensor(c.t), c) for c in cams]
def sample(f, ci, P):
    Rc, tc, c = cams_t[ci]; pc = (P - tc) @ Rc.T
    col = c.cx - c.f * pc[:, 0] / pc[:, 2]; row = c.H - c.cy + c.f * pc[:, 1] / pc[:, 2]
    g = torch.stack([2 * col / c.W - 1, 2 * row / c.H - 1], -1).float()[None, None]
    val = F.grid_sample(gim(f, ci), g, mode='bilinear', align_corners=False)[0, :, 0]
    ins = F.grid_sample(FOVT[ci][None, None], g, mode='nearest', align_corners=False)[0, 0, 0] > 0.5
    return val, ins
def ncc1(x, y):
    x = x - x.mean(); y = y - y.mean(); return (x * y).sum() / torch.sqrt((x * x).sum() * (y * y).sum() + 1e-12)
def rotvec_deg(R): return torch.tensor(np.degrees(Rot.from_matrix(R.detach().numpy()).as_rotvec()))

# committed poses: (R, t) tensors
committed = {n: {start: (torch.tensor(T[n][start][:3, :3]), torch.tensor(T[n][start][:3, 3]))} for n in names}
def extrapolate(n, f):
    c = committed[n]; ks = sorted(c)
    R1, t1 = c[ks[-1]]
    if len(ks) < 2: return Pose(R1, t1, bones[n].centre)
    R0, t0 = c[ks[-2]]; ctr = torch.tensor(bones[n].centre)
    p1, p0 = R1 @ ctr + t1, R0 @ ctr + t0
    dR = R1 @ R0.T; steps = f - ks[-1]
    Rn = torch.tensor(Rot.from_rotvec(Rot.from_matrix(dR.numpy()).as_rotvec() * steps).as_matrix()) @ R1
    pn = p1 + steps * (p1 - p0)
    return Pose(Rn, pn - Rn @ ctr, bones[n].centre)

t0 = time.time(); log = []
for f in range(start + 1, end + 1):
    win = [g for g in range(f, min(f + W, end + 1))]
    base = {(n, g): extrapolate(n, g) for n in names for g in win}
    D = {(n, g): torch.zeros(6, dtype=torch.float64, requires_grad=True) for n in names for g in win}
    def world_pose(n, g):
        if g in committed[n]: return committed[n][g]
        return base[(n, g)].apply(D[(n, g)])
    for res, it, lr in zip((256, 512), ITERS, (0.3, 0.1)):
        opt = torch.optim.Adam(list(D.values()), lr=lr)
        for k in range(it):
            opt.zero_grad(); L = 0
            # DRR term on one random window frame (always include the frame to be committed every other step)
            gs = [win[0]] if k % 2 == 0 else [rng.choice(win)]
            for g in (gs if w_drr > 0 else []):
                rads = [rad(0, g + 1), rad(1, g + 1)]
                with torch.no_grad(): cur = {n: Pose(*world_pose(n, g), bones[n].centre) for n in names}
                for n in names:
                    others = [(bones[m], cur[m]) for m in names if m != n]
                    others += [(bones[m], pose(m, g)) for m in ['1stMet', '2ndMet', '1stPhal', '2ndPhal']
                               if m not in names and not np.isnan(T[m][g, 0, 0])]
                    L = L + w_drr * frame_cost(VIEWS[res], rads, bones[n], base[(n, g)], D[(n, g)], pre,
                                               others=others, joint=bool(others), sim=sim) / len(names)
            for n in names:
                seq = [f - 2, f - 1] + win
                seq = [g for g in seq if g >= start and (g in committed[n] or g in win)]
                ctr = torch.tensor(bones[n].centre, dtype=torch.float64)
                RT = {g: world_pose(n, g) for g in seq}
                # F2F for consecutive pairs ending in the window
                for g in (seq[1:] if w_f2f > 0 else []):
                    if g not in win: continue
                    P0 = PTS[n] @ RT[g - 1][0].T + RT[g - 1][1]; P1 = PTS[n] @ RT[g][0].T + RT[g][1]
                    for ci in range(2):
                        v0, i0 = sample(g - 1, ci, P0); v1, i1 = sample(g, ci, P1); m = i0 & i1
                        L = L + w_f2f * (1 - 0.5 * (ncc1(v0[0][m], v1[0][m]) + ncc1(v0[1][m], v1[1][m]))) / (2 * len(win) * len(names))
                # acceleration penalty (position of bone centre; rotation via small-angle difference of relative rotations)
                pos = {g: RT[g][0] @ ctr + RT[g][1] for g in seq}
                for g in seq[2:]:
                    acc = pos[g] - 2 * pos[g - 1] + pos[g - 2]
                    A = RT[g][0] @ RT[g - 1][0].T; B = RT[g - 1][0] @ RT[g - 2][0].T
                    E = A @ B.T  # ~ identity if constant angular velocity
                    wv = torch.stack([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]]) * 0.5 * 180 / np.pi
                    L = L + w_acc * ((acc ** 2).sum() + (wv ** 2).sum()) / len(names)
            if torch.is_tensor(L) and L.requires_grad: L.backward(); opt.step()
    for n in names:
        with torch.no_grad(): committed[n][f] = base[(n, f)].apply(D[(n, f)])
        if not np.isnan(T[n][f, 0, 0]):
            R, t = committed[n][f]; e = err(n, Pose(R, t, bones[n].centre), f)
            log.append(dict(bone=n, frame=f, mm=e[0], deg=e[1]))
            print(f'fwd {"+".join(names)} {n} @{f} ({f-start} frames from start): {e[0]:.2f}mm {e[1]:.2f}deg  [{time.time()-t0:.0f}s]', flush=True)
tag = '+'.join(names) + f'_{start}_{end}' + (f"_ds{os.environ['DS']}" if os.environ.get('DS') else '')
json.dump({'log': log, 'track': {n: {f: (R.numpy().tolist(), t.numpy().tolist()) for f, (R, t) in committed[n].items()} for n in names},
           'params': dict(w_drr=w_drr, w_f2f=w_f2f, w_acc=w_acc, W=W)}, open(f'fwd_{tag}.json', 'w'))
print('FWDDONE', flush=True)
