"""Keyframe-anchored segment tracking.
Between two trusted keyframes a<b, each bone follows   pose(f) = interp(f) * Delta(f)
with Delta(f) = sum_k c_k sin(k*pi*(f-a)/(b-a))  (6-vector increment about the pivot; zero at both keyframes).
All in-between frames are fitted together (stochastic subset per iteration); several bones can be fitted jointly,
each bone's DRR then also contains the other bones at their current poses (overlap modelled).
Usage: python3 segtrack.py <variant> <a> <b> <bone1,bone2,...> [eval frames]"""
exec(open('/root/work/holdout.py').read().split("kind,vn=")[0])
import json
vn = sys.argv[1]; a, b = int(sys.argv[2]), int(sys.argv[3]); names = sys.argv[4].split(',')
K, LAM, ITERS, BATCH = 3, 0.003, (40, 50), 4
pre = VARIANTS[vn]
frames = list(range(a + 1, b))
basis = {f: torch.tensor([np.sin(k * np.pi * (f - a) / (b - a)) for k in range(1, K + 1)], dtype=torch.float64) for f in frames}
base = {(n, f): interp(n, f, a, b) for n in names for f in frames}
C = {n: torch.zeros(K, 6, dtype=torch.float64, requires_grad=True) for n in names}

def delta(n, f): return basis[f] @ C[n]

def seg_loss(res, fs):
    tot = 0
    for f in fs:
        rads = [rad(0, f + 1), rad(1, f + 1)]
        cur = {}
        with torch.no_grad():
            for n in names:
                R, t = base[(n, f)].apply(delta(n, f)); cur[n] = Pose(R, t, bones[n].centre)
        for n in names:
            others = [(bones[m], cur[m]) for m in names if m != n]
            tot = tot + frame_cost(VIEWS[res], rads, bones[n], base[(n, f)], delta(n, f), pre, others=others, joint=bool(others))
    prior = LAM * sum((C[n] ** 2).sum() for n in names)
    return tot / len(fs) + prior

rng = np.random.default_rng(0); t0 = time.time()
for res, it, lr in zip((256, 512), ITERS, (0.3, 0.1)):
    opt = torch.optim.Adam(list(C.values()), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, it)
    for k in range(it):
        fs = rng.choice(frames, size=min(BATCH, len(frames)), replace=False)
        opt.zero_grad(); L = seg_loss(res, fs); L.backward(); opt.step(); sch.step()
    print(f'  res {res} done, loss {L.item():.4f}, {time.time()-t0:.0f}s', flush=True)

evalf = [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else []
out = {'variant': vn, 'a': a, 'b': b, 'bones': names, 'K': K, 'lam': LAM, 'sec': time.time() - t0, 'eval': [], 'track': {}}
for n in names:
    out['track'][n] = {}
    for f in frames:
        with torch.no_grad(): R, t = base[(n, f)].apply(delta(n, f))
        out['track'][n][f] = (R.numpy().tolist(), t.numpy().tolist())
        if f in evalf and not np.isnan(T[n][f, 0, 0]):
            P = Pose(R, t, bones[n].centre); e0 = err(n, base[(n, f)], f); e1 = err(n, P, f)
            out['eval'].append(dict(bone=n, frame=f, interp_mm=e0[0], interp_deg=e0[1], seg_mm=e1[0], seg_deg=e1[1]))
            print(f'seg:{vn} {"+".join(names)} {a}-{b} {n} @{f}: interp {e0[0]:.2f}mm {e0[1]:.2f}deg -> segment {e1[0]:.2f}mm {e1[1]:.2f}deg', flush=True)
json.dump(out, open(f'seg_{vn}_{"+".join(names)}_{a}_{b}.json', 'w'))
print('SEGDONE', flush=True)
