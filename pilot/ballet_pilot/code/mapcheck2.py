exec(open('/root/work/common.py').read())
from costs import *
views=make_views(cams,512); pre=autoscoper_pre(); d0=torch.zeros(6,dtype=torch.float64)
t0=time.time()
res={0:[],1:[]}
for n in ['1stMet','1stPhal','2ndMet','2ndPhal']:
    for f in [k for k in keyframes(n) if 146<=k<=180]:
        r=[]
        for off in [0,1]:
            with torch.no_grad(): c=frame_cost(views,[rad(0,f+off),rad(1,f+off)],bones[n],pose(n,f),d0,pre).item()
            r.append(c); res[off].append(c)
        print(f'  {n:8s} row {f}: file=row {r[0]:.3f}   file=row+1 {r[1]:.3f}   {"<-- +1 better" if r[1]<r[0] else ""}')
print('mean', np.mean(res[0]), np.mean(res[1]), 'time %.0fs'%(time.time()-t0))
