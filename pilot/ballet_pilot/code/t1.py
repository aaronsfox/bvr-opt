exec(open('setup_synth.py').read())
views=[View(c,512) for c in cams]
t0=time.time()
rads=[synth_rad(v,[(met1,true['met1']),(met2,true['met2'])],i) for i,v in enumerate(views)]
print('synth %.1fs'%(time.time()-t0))
for b,n in [(met1,'1stMet'),(phal1,'1stPhal'),(met2,'2ndMet'),(phal2,'2ndPhal')]:
    print(n,'bbox mm',(b.corners.max(0)-b.corners.min(0)).round(1))
for v in views:
    lo,hi=v.roi(met1,true['met1'].R,true['met1'].t); print('ROI px',(hi-lo).round(0),'render px',((hi-lo)/v.px).round(0))
d=torch.zeros(6,dtype=torch.float64,requires_grad=True)
t0=time.time(); c=cost(views,rads,met1,true['met1'],d); t1=time.time(); c.backward(); t2=time.time()
print('cost at truth %.4f  fwd %.3fs  bwd %.3fs'%(c.item(),t1-t0,t2-t1)); print('grad',d.grad.numpy().round(4))
for e in [1,2,4,8]:
    dd=torch.tensor([e,0,0,0,0,0.],dtype=torch.float64); print('cost dx=%g'%e, cost(views,rads,met1,true['met1'],dd).item())
    dd=torch.tensor([0,0,0,0,0,e],dtype=torch.float64); print('cost rz=%g'%e, cost(views,rads,met1,true['met1'],dd).item())
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,3,figsize=(13,8))
for i,v in enumerate(views):
    lo,hi=v.roi(met1,true['met1'].R,true['met1'].t); lo=lo-60; hi=hi+60
    with torch.no_grad():
        dr=v.render([(met1,true['met1'].R,true['met1'].t)],lo,hi); rd=v.radiograph(rads[i],lo,hi)
    ax[i,0].imshow(rads[i].numpy(),cmap='gray'); ax[i,0].set_title(f'C{i+1} synthetic radiograph')
    ax[i,1].imshow(rd.numpy(),cmap='gray'); ax[i,1].set_title('inverted ROI (1st+2nd met)')
    ax[i,2].imshow(dr.numpy(),cmap='gray'); ax[i,2].set_title('DRR 1st met @512 res')
plt.tight_layout(); plt.savefig('synth.png',dpi=70)
torch.save({'rads':rads},'rads.pt')
