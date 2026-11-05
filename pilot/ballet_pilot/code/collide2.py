exec(open('/root/work/collide.py').read().split("pairs=[")[0].replace("m=ndi.binary_fill_holes(v>0)","m=ndi.binary_fill_holes(ndi.binary_closing(v>THR*v.max(),iterations=2))"))
