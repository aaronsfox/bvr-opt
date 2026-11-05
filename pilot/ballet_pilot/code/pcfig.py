exec(open('/root/work/common.py').read())
import json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
n='1stMet'; f=int(sys.argv[1]); b=bones[n]
# bone point cloud: voxels on strong 3D gradients (cortical outline)
v=b.vol[0,0].numpy(); gz,gy,gx=np.gradient(v,b.sz,b.sy,b.sx); g=np.sqrt(gx**2+gy**2+gz**2)
k,r,c=np.nonzero((v>b.cutoff)&(g>np.quantile(g[v>b.cutoff],0.6)))
sel=np.random.default_rng(0).choice(len(k),size=min(2500,len(k)),replace=False)
P=np.c_[(c[sel]+.5)*b.sx,(b.H-r[sel]-.5)*b.sy,-(k[sel]+.5)*b.sz]
def err(R,t):
    M=T[n][f]; ctr=b.centre
    return np.linalg.norm((R@ctr+t)-(M[:3,:3]@ctr+M[:3,3])), np.degrees(Rot.from_matrix(R@M[:3,:3].T).magnitude())
poses=[('Manual (your tracking)',T[n][f][:3,:3],T[n][f][:3,3],'#00d0ff')]
best=json.load(open('f2f_drr1_f2f2_K3_1stMet+2ndMet_146_166.json'))['track'][n][str(f)]
poses.append(('Best: anchored 146–166,\nDRR grad-corr + frame-to-frame,\n1st+2nd met joint',np.array(best[0]),np.array(best[1]),'#33ff66'))
hc=[x for x in json.load(open('holdout_grad_current.json')) if x['bone']==n and x['frame']==f][0]
poses.append(('Poor: per-frame fit,\nyour Autoscoper filter chain',np.array(hc['R']),np.array(hc['t']),'#ffaa00'))
import os
if f==151 and os.path.exists('pso_1stMet_151.json'):
    p=json.load(open('pso_1stMet_151.json')); poses.append(('Poor: Autoscoper PSO replica,\nyour filter chain',np.array(p['R']),np.array(p['t']),'#ff3355'))
fig,ax=plt.subplots(2,len(poses)+1,figsize=(4.2*(len(poses)+1),9.4))
for ci,cam in enumerate(cams):
    im=rad(ci,f+1).numpy()
    uv_m=cam.project(P@poses[0][1].T+poses[0][2]); ctr=uv_m.mean(0); half=170
    x0,x1,y0,y1=ctr[0]-half,ctr[0]+half,ctr[1]-half*1.25,ctr[1]+half*1.25
    for j,(lab,R,t,col) in enumerate(poses):
        a=ax[ci,j]; a.imshow(im,cmap='gray',vmin=np.percentile(im,1),vmax=np.percentile(im,99.5))
        uv=cam.project(P@R.T+t); a.scatter(uv[:,0],uv[:,1],s=1.2,c=col,alpha=.55,lw=0)
        a.set_xlim(x0,x1); a.set_ylim(y1,y0); a.set_xticks([]); a.set_yticks([])
        e='' if j==0 else '\n%.1f mm, %.1f° from manual'%err(R,t)
        if ci==0: a.set_title(lab+e,fontsize=9)
        if j==0: a.set_ylabel(f'Camera {ci+1}',fontsize=11)
    a=ax[ci,-1]; a.imshow(im,cmap='gray',vmin=np.percentile(im,1),vmax=np.percentile(im,99.5))
    for lab,R,t,col in poses:
        uv=cam.project(P@R.T+t); a.scatter(uv[:,0],uv[:,1],s=0.8,c=col,alpha=.45,lw=0)
    a.set_xlim(x0,x1); a.set_ylim(y1,y0); a.set_xticks([]); a.set_yticks([])
    if ci==0: a.set_title('All overlaid',fontsize=9)
fig.legend([Line2D([],[],marker='o',ls='',color=c_) for *_,c_ in poses],[l.split('\n')[0].rstrip(',') for l,*_ in poses],loc='lower center',ncol=len(poses),fontsize=9,frameon=False)
fig.suptitle(f'1st metatarsal, frame {f} (image file {f+1:04d}): CT bone points projected through the calibration',fontsize=12)
plt.tight_layout(rect=(0,0.03,1,0.96)); plt.savefig(f'pointcloud_1stMet_{f}.png',dpi=90)
print('saved')
