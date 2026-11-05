exec(open('/root/work/common.py').read())
from costs import *; from variants import VARIANTS
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
views=make_views(cams,512)
n,f=sys.argv[1],int(sys.argv[2])
p=pose(n,f)
fig,ax=plt.subplots(4,len(VARIANTS),figsize=(3.2*len(VARIANTS),13))
for ci,v in enumerate(views):
    lo,hi=v.roi(bones[n],p.R,p.t)
    with torch.no_grad():
        d=v.render([(bones[n],p.R,p.t)],lo,hi); r=v.radiograph(rad(ci,f+1),lo,hi)
        for j,(vn,pr) in enumerate(VARIANTS.items()):
            rf,df=pr[:2]; fd=df(d); fr=rf(r).float()
            m=(fd!=0) if len(pr)<3 else (torch.nn.functional.max_pool2d((d>0).float()[None,None],2*pr[2]+1,1,pr[2])[0,0]>0)
            ncc=masked_ncc(fd.float(),fr,m.float()).item()
            ax[2*ci,j].imshow(np.where(m,fr.numpy(),np.nan),cmap='gray'); ax[2*ci,j].set_title(f'C{ci+1} {vn}: radiograph (masked)\nNCC={ncc:.2f}',fontsize=9)
            ax[2*ci+1,j].imshow(np.where(m,fd.numpy(),np.nan),cmap='gray'); ax[2*ci+1,j].set_title(f'C{ci+1} {vn}: DRR',fontsize=9)
for a in ax.flat: a.axis('off')
plt.suptitle(f'{n}, row {f} (file {f+1:04d}) at manual pose'); plt.tight_layout(); plt.savefig(f'var_{n}_{f}.png',dpi=60)
