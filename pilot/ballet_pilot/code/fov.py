exec(open('/root/work/common.py').read())
from costs import *
from scipy.ndimage import binary_erosion
views=make_views(cams,512)
for ci in range(2):
    bg=np.max([tifffile.imread(f'/root/work/rad/c{ci+1}/003_Shoe_Rise_C{ci+1}_00_UND.{k:04d}.tif') for k in range(146,182)],0)/255.
    np.save(f'bgmax_c{ci+1}.npy',bg)
    print(f'C{ci+1}: background(max) histogram:',np.histogram(bg,[0,.05,.1,.2,.4,.6,.8,1.01])[0])
    fov=bg>=0.2; fov_e=binary_erosion(fov,iterations=15); np.save(f'fov_c{ci+1}.npy',fov_e)
    for n in ['1stMet','1stPhal','2ndMet','2ndPhal']:
        fr=[k for k in keyframes(n) if 145<=k<=180]
        worst=0
        for f in fr:
            p=pose(n,f); lo,hi=views[ci].roi(bones[n],p.R,p.t)
            with torch.no_grad(): d=views[ci].render([(bones[n],p.R,p.t)],lo,hi).numpy()>0
            rr,cc=[x.numpy() for x in views[ci].pixel_grid(lo,hi)]
            inside=fov_e[np.clip(rr.astype(int),0,2015),np.clip(cc.astype(int),0,2015)]
            worst=max(worst,1-inside[d].mean()); roiout=1-inside.mean()
        print(f'   {n:8s}: max fraction of bone silhouette outside eroded FOV {worst:.3f}; ROI outside (last frame) {roiout:.3f}')
