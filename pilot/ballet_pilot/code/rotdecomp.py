exec(open('/root/work/common.py').read())
import json, glob
def axes(b):
    v=b.vol[0,0].numpy(); k,r,c=np.nonzero(v>b.cutoff)
    P=np.c_[(c+.5)*b.sx,(b.H-r-.5)*b.sy,-(k+.5)*b.sz]; P=P-P.mean(0)
    w,V=np.linalg.eigh(np.cov(P.T)); return V[:,::-1]
for fn in sorted(glob.glob('holdout_grad_*.json'))+sorted(glob.glob('holdout_pso_*.json')):
    print(fn)
    for r in json.load(open(fn)):
        n,f=r['bone'],r['frame']; M=T[n][f]; ax=axes(bones[n])
        # rotation of refined relative to manual, expressed in bone-local principal axes (deg)
        dR=M[:3,:3].T@np.array(r['R']); rv=np.degrees(Rot.from_matrix(dR).as_rotvec()); comp=ax.T@rv
        c=bones[n].centre; dp=(np.array(r['R'])@c+np.array(r['t']))-(M[:3,:3]@c+M[:3,3]); dpl=ax.T@(M[:3,:3].T@dp)
        print(f"  {n:8s}{f}: rot about long/mid/short {comp[0]:+5.1f} {comp[1]:+5.1f} {comp[2]:+5.1f} deg | shift along long/mid/short {dpl[0]:+5.1f} {dpl[1]:+5.1f} {dpl[2]:+5.1f} mm | cost {r['cost_end']:.3f} vs manual {r['cost_manual']:.3f}")
