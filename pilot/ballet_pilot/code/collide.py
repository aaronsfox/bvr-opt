exec(open('/root/work/common.py').read())
import json
from scipy import ndimage as ndi
SDF={}; SURF={}
for n,b in bones.items():
    v=b.vol[0,0].numpy(); m=ndi.binary_fill_holes(v>0)
    sp=(b.sz,b.sy,b.sx)
    sdf=ndi.distance_transform_edt(~m,sampling=sp)-ndi.distance_transform_edt(m,sampling=sp)
    SDF[n]=sdf
    surf=m & ~ndi.binary_erosion(m)
    k,r,c=np.nonzero(surf)
    SURF[n]=np.c_[(c+.5)*b.sx,(b.H-r-.5)*b.sy,-(k+.5)*b.sz]
def sdf_at(n,Pl):
    b=bones[n]; s=SDF[n]
    idx=np.stack([-Pl[:,2]/b.sz-.5, b.H-Pl[:,1]/b.sy-.5, Pl[:,0]/b.sx-.5])
    return ndi.map_coordinates(s,idx,order=1,mode='nearest')
def pair(nA,MA,nB,MB):
    """min signed distance of A's surface points to B (negative = penetration, mm), and vice versa"""
    out=[]
    for (n1,M1),(n2,M2) in [((nA,MA),(nB,MB)),((nB,MB),(nA,MA))]:
        Pw=SURF[n1]@M1[:3,:3].T+M1[:3,3]; Pl=(Pw-M2[:3,3])@M2[:3,:3]
        out.append(sdf_at(n2,Pl).min())
    return min(out)
def M_of(R,t): M=np.eye(4); M[:3,:3]=np.array(R); M[:3,3]=np.array(t); return M
pairs=[('1stMet','1stPhal'),('2ndMet','2ndPhal'),('1stMet','2ndMet')]
print('Manual keyframes: min surface separation (mm; negative = penetration)')
for a,b2 in pairs:
    fr=[f for f in range(487) if not np.isnan(T[a][f,0,0]) and not np.isnan(T[b2][f,0,0])]
    print(f'  {a}-{b2}:', ' '.join(f'{f}:{pair(a,T[a][f],b2,T[b2][f]):.1f}' for f in fr[:25]))
fm=json.load(open('fwd_1stMet+2ndMet_146_180.json'))['track']; fp=json.load(open('fwd_1stPhal_146_176.json'))['track']
gm=json.load(open('f2f_drr1_f2f2_K3_1stMet+2ndMet_146_176.json'))['track']; gp=json.load(open('f2f_drr1_f2f2_K3_1stPhal_146_176.json'))['track']
for lab,M1,P1 in [('forward-only',fm,fp),('single 30-frame gap',gm,gp)]:
    print(f'\n{lab}: 1stMet-1stPhal and 1stMet-2ndMet separation (mm) by frame')
    s1=[];s2=[]
    for f in range(147,176):
        k=str(f)
        if k in M1['1stMet'] and k in P1['1stPhal']:
            A=M_of(*M1['1stMet'][k]); B=M_of(*P1['1stPhal'][k]); C=M_of(*M1['2ndMet'][k])
            s1.append((f,pair('1stMet',A,'1stPhal',B))); s2.append((f,pair('1stMet',A,'2ndMet',C)))
    print('  MTP1 :',' '.join(f'{f}:{d:.1f}' for f,d in s1[::2]))
    print('  M1-M2:',' '.join(f'{f}:{d:.1f}' for f,d in s2[::2]))
