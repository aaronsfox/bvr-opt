exec(open('/root/work/common.py').read())
from costs import *; from variants import VARIANTS
import pickle
views=make_views(cams,512)
def axes(b):
    P=None
    v=b.vol[0,0].numpy(); k,r,c=np.nonzero(v>b.cutoff)
    P=np.c_[(c+.5)*b.sx,(b.H-r-.5)*b.sy,-(k+.5)*b.sz]; P=P-P.mean(0)
    w,V=np.linalg.eigh(np.cov(P.T)); return V[:,::-1]   # columns: long, mid, short axis (local)
cases=[('1stMet',151),('1stMet',161),('1stMet',171),('1stPhal',156),('1stPhal',166),('2ndMet',156),('2ndMet',166),('2ndPhal',146)]
steps=np.arange(-8,8.01,1.0)
out={}
t0=time.time()
for n,f in cases:
    ax=axes(bones[n]); rads=[rad(0,f+1),rad(1,f+1)]
    for vn,pre in VARIANTS.items():
        S=np.zeros((6,len(steps)))
        for dof in range(6):
            for i,s in enumerate(steps):
                d=np.zeros(6); 
                if dof<3: d[:3]=s*ax[:,dof]
                else: d[3:]=s*ax[:,dof-3]
                with torch.no_grad(): S[dof,i]=frame_cost(views,rads,bones[n],pose(n,f),torch.tensor(d),pre).item()
        out[(n,f,vn)]=S
    print(n,f,'done %.0fs'%(time.time()-t0),flush=True)
pickle.dump((cases,steps,out),open('sweeps.pkl','wb'))
