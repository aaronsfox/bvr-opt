import pickle, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cases,steps,out=pickle.load(open('sweeps.pkl','rb'))
V=['raw','current','edges','lcn']; L=['T long','T mid','T short','R long (axial)','R mid','R short']
print('argmin offset from manual pose (mm or deg) per DOF, and depth (cost at +/-4 minus min, avg)')
summ={v:[] for v in V}
for n,f in cases:
    for v in V:
        S=out[(n,f,v)]; am=steps[S.argmin(1)]
        i4=[np.argmin(abs(steps+4)),np.argmin(abs(steps-4))]
        depth=(S[:,i4].mean(1)-S.min(1))
        summ[v].append((am,depth))
        print(f'{n:8s}{f} {v:8s} argmin {am.astype(int)}  depth@4 {depth.round(3)}')
print()
for v in V:
    am=np.array([a for a,_ in summ[v]]); dp=np.array([d for _,d in summ[v]])
    print(f'{v:8s} median |argmin| T {np.median(abs(am[:,:3])):.1f} mm, R {np.median(abs(am[:,3:])):.1f} deg; frac DOFs min within 1: {(abs(am)<=1).mean():.2f}; median depth {np.median(dp):.3f}; rel depth (depth/min) {np.median(dp/np.array([[out[(n,f,v)].min()]*6 for n,f in cases])):.3f}')
fig,ax=plt.subplots(len(cases),6,figsize=(18,2.3*len(cases)),sharex=True)
for i,(n,f) in enumerate(cases):
    for j in range(6):
        for v,c in zip(V,['0.6','C0','C2','C3']):
            S=out[(n,f,v)][j]; ax[i,j].plot(steps,(S-S.min())/(S.max()-S.min()+1e-9),color=c,label=v)
        ax[i,j].axvline(0,color='k',lw=.5)
        if i==0: ax[i,j].set_title(L[j])
        if j==0: ax[i,j].set_ylabel(f'{n}\n{f}')
ax[0,0].legend(fontsize=7); plt.tight_layout(); plt.savefig('sweeps.png',dpi=55)
