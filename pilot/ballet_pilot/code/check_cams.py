import sys; sys.path.insert(0,'.')
import numpy as np, tifffile, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from ascam import MayaCam1
A='/mnt/user-data/attachments/de80d863-0bfc-5776-b7cf-80da25e9c8d8/'
cams=[MayaCam1(A+'5e7fc9b4-Lego_1_C1_MayaCam.csv'),MayaCam1(A+'2ff3bea2-Lego_1_C2_MayaCam.csv')]
pts=np.array([[float(x) for x in l.strip().strip(',').split(',')] for l in open('xma/CalibrationObject/Framespec_Lego.csv').read().splitlines()[1:] if l.strip()])
print('cube pts',len(pts),'extent',pts.min(0),pts.max(0))
ctr=pts.mean(0)
for i,c in enumerate(cams):
    d=ctr-c.t; d/=np.linalg.norm(d)
    print(f'C{i+1}: pos {c.t.round(1)}, dist to cube centre {np.linalg.norm(ctr-c.t):.0f}, angle look-vs-centre {np.degrees(np.arccos(c.look@d)):.2f} deg, f={c.f:.0f}px, FOV {np.degrees(2*np.arctan(c.W/2/c.f)):.1f} deg, cx,cy={c.cx:.0f},{c.cy:.0f}')
print('inter-beam angle %.1f deg'%np.degrees(np.arccos(abs(cams[0].look@cams[1].look))))
fig,ax=plt.subplots(1,2,figsize=(14,7))
for i,c in enumerate(cams):
    im=tifffile.imread(f'xma/Camera {i+1}/Lego_1_C{i+1}.tif'); print('image',im.shape,im.dtype)
    if im.ndim==3: im=im[...,0]
    uv=c.project(pts)
    ax[i].imshow(im,cmap='gray'); ax[i].plot(uv[:,0],uv[:,1],'r+',ms=8); ax[i].set_title(f'C{i+1}: Lego cube points projected with Autoscoper camera model'); ax[i].axis('off')
plt.tight_layout(); plt.savefig('cam_check.png',dpi=80)
