s=open('costs.py').read()
s=s.replace('''    rad_fn, drr_fn = pre
''','''    rad_fn, drr_fn = pre[:2]
    dil = pre[2] if len(pre) > 2 else None      # None: Autoscoper mask (filtered DRR != 0)
''')
s=s.replace('''        own = drr_fn(v.render([(bone, R, t)], lo, hi))
        if joint and others:
            drr = drr_fn(v.render([(bone, R, t)] + [(b, p.R, p.t) for b, p in others], lo, hi))
        else:
            drr = own
        mask = (own.detach() != 0).to(drr.dtype)''','''        raw = v.render([(bone, R, t)], lo, hi)
        own = drr_fn(raw)
        if joint and others:
            drr = drr_fn(v.render([(bone, R, t)] + [(b, p.R, p.t) for b, p in others], lo, hi))
        else:
            drr = own
        if dil is None:
            mask = (own.detach() != 0).to(drr.dtype)
        else:   # silhouette of the unfiltered DRR, dilated by `dil` render pixels
            m = (raw.detach() > 0).float()[None, None]
            mask = (F.max_pool2d(m, 2 * dil + 1, 1, dil)[0, 0] if dil > 0 else m[0, 0]).to(drr.dtype)''')
s=s.replace('import math, torch\n','import math, torch\nimport torch.nn.functional as F\n')
open('costs.py','w').write(s)
v=open('variants.py').read()
v=v.replace("'edges':    (lambda r:edges(r), lambda d:edges(d)),","'edges':    (lambda r:edges(r), lambda d:edges(d), 3),")
v=v.replace("'lcn':      (lambda r:lcn(gaussian(r,1.0)), lambda d:lcn(gaussian(d,1.0))),","'lcn':      (lambda r:lcn(gaussian(r,1.0)), lambda d:lcn(gaussian(d,1.0)), 3),")
open('variants.py','w').write(v)
s=open('varfig.py').read()
s=s.replace("for j,(vn,(rf,df)) in enumerate(VARIANTS.items()):\n            fd=df(d); fr=rf(r).float(); m=(fd!=0)",
"for j,(vn,pr) in enumerate(VARIANTS.items()):\n            rf,df=pr[:2]; fd=df(d); fr=rf(r).float()\n            m=(fd!=0) if len(pr)<3 else (torch.nn.functional.max_pool2d((d>0).float()[None,None],2*pr[2]+1,1,pr[2])[0,0]>0)")
s=s.replace("ncc=masked_ncc(fd,fr,m.float()).item()","ncc=masked_ncc(fd.float(),fr,m.float()).item()")
open('varfig.py','w').write(s)
