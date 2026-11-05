exec(open('/root/work/holdout.py').read().split("kind,vn=")[0])
pre=VARIANTS['edges']
for n,f,a,b in [('1stMet',161,156,166),('1stMet',151,146,156)]:
    P0=interp(n,f,a,b); rads=[rad(0,f+1),rad(1,f+1)]
    fn=lambda res,x: frame_cost(VIEWS[res],rads,bones[n],P0,x,pre)
    for lr in [(0.3,0.15,0.05),(0.15,0.08,0.03),(0.1,0.1,0.1)]:
        for resl in [(128,256,512),(256,512,512)]:
            t0=time.time(); d,c,ne=gradient_refine(fn,resolutions=resl,lr=lr); P1=P0.compose(d)
            print(n,f,lr,resl,'start %.2fmm %.2fdeg'%err(n,P0,f),'-> %.2fmm %.2fdeg'%err(n,P1,f),'cost %.4f'%c,'%.0fs'%(time.time()-t0),flush=True)
