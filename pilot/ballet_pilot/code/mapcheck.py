exec(open('/root/work/common.py').read())
views=[View(c,512) for c in cams]
d0=torch.zeros(6,dtype=torch.float64)
print('raw-intensity cost (sum 1-NCC, 2 cams) at keyframe poses, image file = row + offset')
for n in ['1stMet','1stPhal','2ndMet','2ndPhal']:
    for f in [k for k in keyframes(n) if 146<=k<=180]:
        r=[]
        for off in [0,1]:
            if not (146<=f+off<=181): r.append(np.nan); continue
            with torch.no_grad(): r.append(cost(views,[rad(0,f+off),rad(1,f+off)],bones[n],pose(n,f),d0).item())
        print(f'  {n:8s} row {f}: offset0 {r[0]:.3f}  offset+1 {r[1]:.3f}')
