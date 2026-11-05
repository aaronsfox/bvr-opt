import sys; sys.path.insert(0,'/root/work')
import numpy as np, torch, tifffile, glob, time
from ascam import MayaCam1
from asdrr import *
from scipy.spatial.transform import Rotation as Rot
A='/mnt/user-data/attachments/de80d863-0bfc-5776-b7cf-80da25e9c8d8/'
cams=[MayaCam1(A+'5e7fc9b4-Lego_1_C1_MayaCam.csv'),MayaCam1(A+'2ff3bea2-Lego_1_C2_MayaCam.csv')]
VOL={'1stMet':'c69e78b7-1stMet.tif','1stPhal':'b218988a-1stPhal.tif','2ndMet':'47265cba-2ndMet.tif','2ndPhal':'5cc0aace-2ndPhal.tif'}
TRA={'1stMet':'0fb7b310-1stMet_Tracking.tra','1stPhal':'66799824-1stPhal_Tracking.tra','2ndMet':'7db8de1d-2ndMet_Tracking.tra','2ndPhal':'3594c906-2ndPhal_Initial.tra'}
bones={n:Bone(A+f) for n,f in VOL.items()}
T={n:np.genfromtxt(A+f,delimiter=',').reshape(-1,4,4) for n,f in TRA.items()}
def keyframes(n): return np.nonzero(~np.isnan(T[n][:,0,0]))[0]
def pose(n,f): M=T[n][f]; return Pose(M[:3,:3],M[:3,3],bones[n].centre)
_cache={}
def rad(cam,filenum):
    k=(cam,filenum)
    if k not in _cache:
        im=torch.tensor(tifffile.imread(f'/root/work/rad/c{cam+1}/003_Shoe_Rise_C{cam+1}_00_UND.{filenum:04d}.tif').astype(np.float32)/255.)
        ds=int(__import__('os').environ.get('DS','0'))   # optional Williams-style bicubic downsampling of the input images
        if ds: im=torch.nn.functional.interpolate(im[None,None],size=(ds,ds),mode='bicubic',antialias=True,align_corners=False)[0,0].clamp(0,1)
        _cache[k]=im
    return _cache[k]
