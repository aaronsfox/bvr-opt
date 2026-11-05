import sys; sys.argv=['x','0','1','166','176','1stMet,2ndMet','171']
src=open('/root/work/segtrack2.py').read().split("t0 = time.time(); hist")[0]
exec(src)
t=time.time(); L0=f2f_loss().item(); print('f2f loss on interpolated path %.4f (%.2fs)'%(L0,time.time()-t))
for dof,lab in [(0,'x 2mm'),(1,'y 2mm'),(2,'z 2mm'),(3,'rx 3deg'),(4,'ry 3deg'),(5,'rz 3deg')]:
    with torch.no_grad():
        C['1stMet'].zero_(); C['1stMet'][0,dof]=2.0 if dof<3 else 3.0
        # first sine bump: shift peaks mid-gap; f2f sees the change in motion between frames
        print(f'  1stMet bump {lab}: f2f {f2f_loss().item():.4f}')
        C['1stMet'][1,dof]=C['1stMet'][0,dof]; C['1stMet'][0,dof]=0
        print(f'  1stMet 2nd-harmonic {lab}: f2f {f2f_loss().item():.4f}')
C['1stMet'].data.zero_()
t=time.time(); L=f2f_loss(); L.backward(); print('backward %.2fs'%(time.time()-t))
