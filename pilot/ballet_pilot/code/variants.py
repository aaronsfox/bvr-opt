import torch, torch.nn.functional as F
from filters import apply_chain, gaussian, sobel, _gauss_kernel, _conv
from costs import VIE
def _blur(x,s):
    r=int(3*s); ax=torch.arange(-r,r+1,dtype=torch.float64); k=torch.exp(-ax**2/(2*s*s)); k=(k[:,None]*k[None,:]); k=k/k.sum()
    return _conv(x,k)
def lcn(x, s=6.0, eps=0.02):
    m=_blur(x,s); v=_blur((x-m)**2,s)
    return (x-m)/torch.sqrt(v+eps**2)
def edges(x, pre=1.0):
    return sobel(gaussian(x,pre),scale=1.0,blend=0.0)
VARIANTS={
 'raw':      (lambda r:r, lambda d:d),
 'current':  (lambda r:apply_chain(r,VIE['rad']), lambda d:apply_chain(d,VIE['drr'])),
 'edges':    (lambda r:edges(r), lambda d:edges(d), 3),
 'lcn':      (lambda r:lcn(gaussian(r,1.0)), lambda d:lcn(gaussian(d,1.0)), 3),
}

# ---- additional similarity measures / transforms ----
from asdrr import masked_ncc
def lograd(x):            # x is the inverted radiograph (1 - I) -> line-integral scale -log(I)
    return -torch.log((1 - x).clamp_min(1.0 / 255))
def _sobel_xy(x):
    p = F.pad(x[None, None], (1, 1, 1, 1), mode='replicate')[0, 0]
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]) - (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
    return gx, gy
def grad_corr(d, r, m):   # signed gradient correlation (mean of NCC of x- and y-gradients)
    dx, dy = _sobel_xy(d); rx, ry = _sobel_xy(r.to(d.dtype))
    return 0.5 * (masked_ncc(dx, rx, m) + masked_ncc(dy, ry, m))
def local_ncc(d, r, m, s=4.0):   # Gaussian-windowed NCC, averaged over the mask
    r = r.to(d.dtype); w = _blur(m, s) + 1e-6
    md = _blur(d * m, s) / w; mr = _blur(r * m, s) / w
    vd = _blur(d * d * m, s) / w - md ** 2; vr = _blur(r * r * m, s) / w - mr ** 2
    cv = _blur(d * r * m, s) / w - md * mr
    lncc = cv / torch.sqrt(vd.clamp_min(0) * vr.clamp_min(0) + 1e-8)
    return (lncc * m).sum() / m.sum()
g1 = lambda x: gaussian(x, 1.0)
COMBOS = {   # name: (pre, sim)
  'log+lcn':      ((lambda r: lcn(gaussian(lograd(r) / 3, 1.0)), lambda d: lcn(gaussian(d, 1.0)), 3), None),
  'log+gradcorr': ((lambda r: _blur(lograd(r), 1.0), lambda d: _blur(d, 1.0), 3), grad_corr),
  'gradcorr':     ((lambda r: _blur(r, 1.0), lambda d: _blur(d, 1.0), 3), grad_corr),
  'log+localncc': ((lambda r: _blur(lograd(r), 1.0), lambda d: _blur(d, 1.0), 3), local_ncc),
  'edges':        (VARIANTS['edges'], None),
}
