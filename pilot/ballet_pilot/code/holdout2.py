# per-frame hold-out with joint rendering of the other tracked bones (at manual poses) for new similarity measures
exec(open('/root/work/holdout.py').read().split("kind,vn=")[0])
from variants import COMBOS
import costs; costs.USE_FOV = True
import json
names=sys.argv[1].split(','); out=[]
for cn in names:
    pre,sim=COMBOS[cn]
    for n,f,a,b in CASES:
        P0=interp(n,f,a,b); rads=[rad(0,f+1),rad(1,f+1)]
        others=[(bones[m],pose(m,f)) for m in ['1stMet','2ndMet','1stPhal','2ndPhal'] if m!=n and not np.isnan(T[m][f,0,0])]
        fn=lambda res,x: frame_cost(VIEWS[res],rads,bones[n],P0,x,pre,others=others,joint=True,sim=sim)
        t0=time.time(); d,c,ne=gradient_refine(fn); P1=P0.compose(d)
        with torch.no_grad(): cm=frame_cost(VIEWS[512],rads,bones[n],pose(n,f),torch.zeros(6,dtype=torch.float64),pre,others=others,joint=True,sim=sim).item()
        e0,e1=err(n,P0,f),err(n,P1,f)
        out.append(dict(combo=cn,bone=n,frame=f,start_mm=e0[0],start_deg=e0[1],end_mm=e1[0],end_deg=e1[1],cost_end=c,cost_manual=cm,R=P1.R.numpy().tolist(),t=P1.t.numpy().tolist()))
        print(f"h2:{cn} {n} {f}: start {e0[0]:.2f}mm {e0[1]:.2f}deg -> end {e1[0]:.2f}mm {e1[1]:.2f}deg | cost end {c:.4f} manual {cm:.4f} | {time.time()-t0:.0f}s",flush=True)
        json.dump(out,open('holdout2_'+'_'.join(names).replace('+','p')+'.json','w'))
