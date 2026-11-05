exec(open('/root/work/common.py').read())
from costs import *; from variants import VARIANTS; from optim import *
from scipy.spatial.transform import Slerp
import json, sys
METHOD=sys.argv[1] if len(sys.argv)>1 else "grad:edges"
CASES=[('1stMet',151,146,156),('1stMet',156,151,161),('1stMet',161,156,166),('1stMet',166,161,171),('1stMet',171,166,176),
       ('2ndMet',171,166,176),('2ndMet',176,171,181),('1stPhal',156,146,166),('1stPhal',166,156,176)]
if len(sys.argv)>2 and __name__=="__main__": CASES=[c for c in CASES if f'{c[0]}:{c[1]}' in sys.argv[2].split(',')]
VIEWS={r:make_views(cams,r) for r in (128,256,512)}
def interp(n,f,a,b):
    w=(f-a)/(b-a); Ma,Mb=T[n][a],T[n][b]; c=bones[n].centre
    pa,pb=Ma[:3,:3]@c+Ma[:3,3],Mb[:3,:3]@c+Mb[:3,3]
    R=Slerp([0,1],Rot.from_matrix([Ma[:3,:3],Mb[:3,:3]]))([w]).as_matrix()[0]
    p=(1-w)*pa+w*pb
    return Pose(R,p-R@c,c)
def err(n,P,f):
    M=T[n][f]; c=bones[n].centre
    dt=np.linalg.norm((P.R.numpy()@c+P.t.numpy())-(M[:3,:3]@c+M[:3,3]))
    da=np.degrees(Rot.from_matrix(P.R.numpy()@M[:3,:3].T).magnitude()); return dt,da
kind,vn=METHOD.split(':'); pre=VARIANTS[vn]
out=[]
for n,f,a,b in CASES:
    P0=interp(n,f,a,b); rads=[rad(0,f+1),rad(1,f+1)]
    t0=time.time()
    if kind=='grad':
        fn=lambda res,x: frame_cost(VIEWS[res],rads,bones[n],P0,x,pre)
        d,c,ne=gradient_refine(fn)
    else:
        def fnp(x):
            with torch.no_grad(): return frame_cost(VIEWS[512],rads,bones[n],P0,torch.tensor(x),pre).item()
        d,c,ne=autoscoper_pso_nm(fnp,max_epochs=25,max_stall=10,seed=f)
    P1=P0.compose(d)
    with torch.no_grad(): cm=frame_cost(VIEWS[512],rads,bones[n],pose(n,f),torch.zeros(6,dtype=torch.float64),pre).item()
    e0,e1=err(n,P0,f),err(n,P1,f)
    r=dict(bone=n,frame=f,gap=b-a,start_mm=e0[0],start_deg=e0[1],end_mm=e1[0],end_deg=e1[1],cost_end=c,cost_manual=cm,evals=ne,sec=time.time()-t0,
           R=P1.R.numpy().tolist(),t=P1.t.numpy().tolist())
    out.append(r)
    print(f"{METHOD} {n} {f}: start {e0[0]:.2f}mm {e0[1]:.2f}deg -> end {e1[0]:.2f}mm {e1[1]:.2f}deg | cost end {c:.4f} manual {cm:.4f} | {ne} evals {time.time()-t0:.0f}s",flush=True)
json.dump(out,open(f'holdout_{kind}_{vn}.json','w'))
