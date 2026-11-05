exec(open('tra.py').read())
from ascam import MayaCam1
from scipy.spatial.transform import Rotation as Rot
import tifffile
vols={'1stMet':'c69e78b7-1stMet.tif','1stPhal':'b218988a-1stPhal.tif','2ndMet':'47265cba-2ndMet.tif','2ndPhal':'5cc0aace-2ndPhal.tif'}
sx,sy,sz=0.625,0.625,1.25
cams=[MayaCam1(A+'5e7fc9b4-Lego_1_C1_MayaCam.csv'),MayaCam1(A+'2ff3bea2-Lego_1_C2_MayaCam.csv')]
info={}
for n,f in vols.items():
    v=tifffile.imread(A+f); D,H,W=v.shape; k,r,c=np.nonzero(v)
    P=np.c_[(c+.5)*sx,(H-r-.5)*sy,-(k+.5)*sz]          # Autoscoper local frame
    info[n]=dict(P=P[::7],cen=P.mean(0),D=D,H=H,W=W)
T={n:load(n).reshape(-1,4,4) for n in files}
for n in files:
    M=T[n]; idx=np.nonzero(~np.isnan(M[:,0,0]))[0]
    R=M[idx,:3,:3]; orth=np.abs(np.einsum('nij,nkj->nik',R,R)-np.eye(3)).max()
    print(f'{n}: {len(idx)} tracked, frames {idx.tolist()}  (max orthonormality error {orth:.1e})')
# relative-position sanity at a shared frame: closest-surface distance met1 vs phal1
for a,b in [('1stMet','1stPhal'),('2ndMet','2ndPhal'),('1stMet','2ndMet')]:
    fa=set(np.nonzero(~np.isnan(T[a][:,0,0]))[0]); fb=set(np.nonzero(~np.isnan(T[b][:,0,0]))[0]); sh=sorted(fa&fb)
    if not sh: print(a,b,'no shared frame'); continue
    f=sh[0]
    for lab,fn in [('M@p (col-vector)',lambda M,P:P@M[:3,:3].T+M[:3,3]),('p@M (row-vector)',lambda M,P:P@M[:3,:3]+M[:3,3])]:
        Pa=fn(T[a][f],info[a]['P']); Pb=fn(T[b][f],info[b]['P'])
        from scipy.spatial import cKDTree
        d=cKDTree(Pa).query(Pb)[0]
        print(f'  frame {f} {a}-{b} [{lab}]: min surface gap {d.min():.1f} mm, centres {np.linalg.norm(Pa.mean(0)-Pb.mean(0)):.1f} mm apart')
print()
def world(M,p): return M[:3,:3]@p+M[:3,3]
for n in ['1stMet','1stPhal','2ndMet']:
    M=T[n]; idx=np.nonzero(~np.isnan(M[:,0,0]))[0]; c=info[n]['cen']
    print(f'{n}: gap -> centroid move (mm), rotation (deg), rate per frame')
    rows=[]
    for a,b in zip(idx[:-1],idx[1:]):
        dt=np.linalg.norm(world(M[b],c)-world(M[a],c))
        ang=np.degrees(Rot.from_matrix(M[b,:3,:3]@M[a,:3,:3].T).magnitude())
        rows.append((a,b,dt,ang))
    for a,b,dt,ang in rows:
        flag=' <-- fast' if (dt/(b-a)>0.4 or ang/(b-a)>0.8) else ''
        print(f'   {a:3d}-{b:3d} ({b-a:2d} fr): {dt:5.1f} mm {ang:5.1f} deg   {dt/(b-a):.2f} mm/fr {ang/(b-a):.2f} deg/fr{flag}')
# relative MTP angle over time
print('\n1st MTP (phal rel. met) rotation from first shared frame:')
f1=sorted(set(np.nonzero(~np.isnan(T['1stMet'][:,0,0]))[0])&set(np.nonzero(~np.isnan(T['1stPhal'][:,0,0]))[0]))
R0=T['1stMet'][f1[0],:3,:3].T@T['1stPhal'][f1[0],:3,:3]
print('  ',' '.join(f'{f}:{np.degrees(Rot.from_matrix(R0.T@(T["1stMet"][f,:3,:3].T@T["1stPhal"][f,:3,:3])).magnitude()):.0f}' for f in f1))
# field of view check
print('\nprojected 1stMet centroid (px) per camera at keyframes 88/146/216/341:')
for f in [88,146,216,341]:
    p=world(T['1stMet'][f],info['1stMet']['cen'])
    print(f'  {f}: ', ['(%.0f,%.0f)'%tuple(c.project(p[None])[0]) for c in cams])
