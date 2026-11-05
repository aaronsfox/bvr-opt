exec(open('/root/work/holdout.py').read().split("kind,vn=")[0])
import json
n,f,a,b='1stMet',151,146,156
P0=interp(n,f,a,b); rads=[rad(0,f+1),rad(1,f+1)]; pre=VARIANTS['current']
def fnp(x):
    with torch.no_grad(): return frame_cost(VIEWS[512],rads,bones[n],P0,torch.tensor(x),pre).item()
d,c,ne=autoscoper_pso_nm(fnp,max_epochs=25,max_stall=10,seed=f)
P1=P0.compose(d); e=err(n,P1,f); print('PSO 151',e,c,flush=True)
json.dump(dict(R=P1.R.numpy().tolist(),t=P1.t.numpy().tolist(),err=e,cost=c),open('pso_1stMet_151.json','w'))
