# 2nd met hold-out with the 1st met rendered too (fixed at its manual pose): does modelling overlap remove the false minimum?
exec(open('/root/work/holdout.py').read().split("kind,vn=")[0])
import json
out=[]
for vn in sys.argv[1].split(','):
    pre=VARIANTS[vn]
    for n,f,a,b in [c for c in CASES if c[0]=='2ndMet']:
        P0=interp(n,f,a,b); rads=[rad(0,f+1),rad(1,f+1)]
        others=[(bones['1stMet'],pose('1stMet',f))] + ([(bones['2ndPhal'],pose('2ndPhal',f))] if not np.isnan(T['2ndPhal'][f,0,0]) else []) \
               + ([(bones['1stPhal'],pose('1stPhal',f))] if not np.isnan(T['1stPhal'][f,0,0]) else [])
        fn=lambda res,x: frame_cost(VIEWS[res],rads,bones[n],P0,x,pre,others=others,joint=True)
        t0=time.time(); d,c,ne=gradient_refine(fn); P1=P0.compose(d)
        with torch.no_grad(): cm=frame_cost(VIEWS[512],rads,bones[n],pose(n,f),torch.zeros(6,dtype=torch.float64),pre,others=others,joint=True).item()
        e0,e1=err(n,P0,f),err(n,P1,f)
        out.append(dict(variant=vn,bone=n,frame=f,start_mm=e0[0],start_deg=e0[1],end_mm=e1[0],end_deg=e1[1],cost_end=c,cost_manual=cm,R=P1.R.numpy().tolist(),t=P1.t.numpy().tolist()))
        print(f"joint:{vn} {n} {f}: start {e0[0]:.2f}mm {e0[1]:.2f}deg -> end {e1[0]:.2f}mm {e1[1]:.2f}deg | cost end {c:.4f} manual {cm:.4f} | {time.time()-t0:.0f}s",flush=True)
json.dump(out,open('holdout_joint.json','w'))
