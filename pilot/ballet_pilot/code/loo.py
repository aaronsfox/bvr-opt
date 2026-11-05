exec(open('analyse.py').read().split("print()\n")[0])
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
def world(M,p): return M[:3,:3]@p+M[:3,3]
for n in ['1stMet','1stPhal','2ndMet']:
    M=T[n]; idx=np.nonzero(~np.isnan(M[:,0,0]))[0]; c=info[n]['cen']
    print(f'{n}: leave-one-out interpolation error at each interior keyframe (linear+slerp | cubic)')
    out=[]
    for i in range(1,len(idx)-1):
        keep=np.delete(idx,i); f=idx[i]
        a,b=keep[i-1],keep[i]
        if b-a>20: continue
        C=np.array([world(M[k],c) for k in keep]); rots=Rot.from_matrix(M[keep,:3,:3])
        sl=Slerp(keep,rots)
        lin=C[i-1]+(C[i]-C[i-1])*(f-a)/(b-a)
        cub=CubicSpline(keep,C)(f) if len(keep)>3 else lin
        true=world(M[f],c); Rt=Rot.from_matrix(M[f,:3,:3])
        el=np.linalg.norm(lin-true); ec=np.linalg.norm(cub-true); er=np.degrees((sl([f])[0]*Rt.inv()).magnitude())
        out.append((f,b-a,el,ec,er))
        print(f'   frame {f:3d} (gap {b-a:2d}): pos err lin {el:4.1f} / cubic {ec:4.1f} mm, rot err {er:4.1f} deg')
    o=np.array(out)[:,1:]
    for g in sorted(set(o[:,0])):
        s=o[o[:,0]==g]; print(f'   -> gap {int(g)}: median pos lin {np.median(s[:,1]):.1f} cubic {np.median(s[:,2]):.1f} mm, rot {np.median(s[:,3]):.1f} deg (n={len(s)})')
