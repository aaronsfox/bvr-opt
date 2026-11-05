import sys; sys.path.insert(0,'.')
import numpy as np, torch, time
from ascam import MayaCam1
from asdrr import *
A='/mnt/user-data/attachments/de80d863-0bfc-5776-b7cf-80da25e9c8d8/'
cams=[MayaCam1(A+'5e7fc9b4-Lego_1_C1_MayaCam.csv'),MayaCam1(A+'2ff3bea2-Lego_1_C2_MayaCam.csv')]
met1=Bone(A+'c69e78b7-1stMet.tif'); met2=Bone(A+'47265cba-2ndMet.tif')
phal1=Bone(A+'b218988a-1stPhal.tif'); phal2=Bone(A+'5cc0aace-2ndPhal.tif')
def place(bone, centre_world, rv_deg):
    R=rotvec_to_matrix(torch.tensor(rv_deg,dtype=torch.float64))
    t=torch.tensor(centre_world,dtype=torch.float64)-R@torch.tensor(bone.centre)
    return Pose(R,t,bone.centre)
C=np.array([96.,96.,86.])
true={'met1':place(met1,C,[0,0,0]),'met2':place(met2,C+[0,-17,4],[0,0,6])}
def synth_rad(view, items, seed, res_px=2.0):
    """Synthetic raw radiograph (full image): soft-tissue blob + bones + noise; bone = dark."""
    rng=np.random.default_rng(seed); cam=view.cam
    W=cam.W; inv=np.zeros((W,W),np.float32)
    # bones rendered at res_px over the union ROI
    v2=View(cam,render_res=int(W/res_px))
    los,his=zip(*[v2.roi(b,p.R,p.t) for b,p in items]); lo=np.min(los,0)-40; hi=np.max(his,0)+40
    with torch.no_grad(): d=v2.render([(b,p.R,p.t) for b,p in items],lo,hi).numpy()
    yy,xx=np.mgrid[0:W,0:W]
    from scipy.ndimage import zoom
    d=zoom(d,res_px,order=1); r0,c0=int(lo[1]),int(lo[0])
    inv[r0:r0+d.shape[0],c0:c0+d.shape[1]]=d[:W-r0,:W-c0] if False else d[:min(d.shape[0],W-r0),:min(d.shape[1],W-c0)]
    inv=inv/ (inv.max()+1e-9)*0.5
    ctr=np.mean([cam.project((p.R@torch.tensor(b.centre)+p.t).numpy()[None])[0] for b,p in items],0)
    soft=0.25*np.exp(-(((xx-ctr[0])/350)**2+((yy-ctr[1])/250)**2))
    inv=inv+soft+rng.normal(0,0.02,inv.shape).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    inv=gaussian_filter(inv,1.0)
    return torch.tensor(np.clip(1-inv,0,1),dtype=torch.float32)
