exec(open('/root/work/common.py').read())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
f=int(sys.argv[1]); views=[View(c,1024) for c in cams]
cols={'1stMet':'r','1stPhal':'y','2ndMet':'c','2ndPhal':'m'}
fig,ax=plt.subplots(2,3,figsize=(18,12))
for ci,v in enumerate(views):
    ax[ci,0].imshow(rad(ci,f).numpy(),cmap='gray'); ax[ci,0].set_title(f'C{ci+1} file {f:04d} (full)')
    # union ROI
    los,his=[],[]
    for n in cols:
        if not np.isnan(T[n][f,0,0]): p=pose(n,f); lo,hi=v.roi(bones[n],p.R,p.t); los.append(lo); his.append(hi)
    lo=np.min(los,0)-80; hi=np.max(his,0)+80
    ax[ci,0].add_patch(plt.Rectangle(lo,*(hi-lo),fill=False,ec='r'))
    for j,off in enumerate([0,1]):
        a=ax[ci,j+1]; im=v.radiograph(rad(ci,f+off) if f+off<=181 else rad(ci,f),lo,hi)
        a.imshow(1-im.numpy(),cmap='gray')
        for n,c in cols.items():
            if np.isnan(T[n][f,0,0]): continue
            p=pose(n,f)
            with torch.no_grad(): d=v.render([(bones[n],p.R,p.t)],lo,hi).numpy()
            a.contour(d>0,[0.5],colors=c,linewidths=1)
        a.set_title(f'C{ci+1}: row {f} poses on file {f+off:04d}'); a.axis('off')
plt.tight_layout(); plt.savefig(f'overlay_{f}.png',dpi=55)
