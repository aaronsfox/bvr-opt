"""PyTorch ports of Autoscoper's OpenCL image filters (libautoscoper/src/gpu/opencl/kernel/*.cl).
All take/return (H,W) float tensors and are differentiable."""
import math, torch, torch.nn.functional as F


def _conv(img, k):
    r = (k.shape[0] - 1) // 2
    return F.conv2d(img[None, None], k[None, None].to(img.dtype), padding=r)[0, 0]  # zero outside, like the kernel


def _gauss_kernel(radius):
    # Autoscoper: size = 2*int(3*radius)+1, weight = exp(-d^2 / (2*radius)) i.e. variance = radius
    fr = int(3 * radius)
    if fr == 0:
        return None
    ax = torch.arange(-fr, fr + 1, dtype=torch.float64)
    k = torch.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2.0 * radius))
    return k / k.sum()


def sobel(img, scale=1.0, blend=0.5):
    p = F.pad(img[None, None], (1, 1, 1, 1), mode='replicate')[0, 0]
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]) - (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
    grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    if blend < 0.5:
        out = scale * grad + 2.0 * blend * img
    else:
        out = 2.0 * (1.0 - blend) * scale * grad + img
    return out.clamp(0, 1)


def contrast(img, alpha=1.0, beta=1.0, size=3):
    a = F.avg_pool2d(img[None, None], size, stride=1, padding=size // 2, count_include_pad=False)[0, 0]
    out = a.clamp_min(1e-6) ** (alpha - beta) * img.clamp_min(0) ** beta
    return torch.where(a > 0.01, out, torch.zeros_like(out))


def gaussian(img, radius=1.0):
    k = _gauss_kernel(radius)
    return img.clamp(0, 1) if k is None else _conv(img, k).clamp(0, 1)


def sharpen(img, radius=1.0, contrast_=1.0, threshold=0.0):
    k = _gauss_kernel(radius)
    if k is None:
        return img
    blur = _conv(img, k)
    out = (img + contrast_ * (img - blur)).clamp(0, 1)
    return torch.where((img - blur).abs() > threshold, out, img)


def parse_vie(path):
    """Returns {'rad': [(name, params)], 'drr': [...], 'renderer': {...}} from a .vie settings file."""
    out = {'rad': [], 'drr': [], 'renderer': {}}
    cur, filt = None, None
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line == 'RadFilters_begin': cur = 'rad'
        elif line == 'DrrFilters_begin': cur = 'drr'
        elif line == 'DrrRenderer_begin': cur = 'renderer'
        elif line.endswith('_end') and line[:-4] in ('RadFilters', 'DrrFilters', 'DrrRenderer'): cur = None
        elif line.endswith('_begin'): filt = (line[:-6], {}); out[cur].append(filt)
        elif line.endswith('_end'): filt = None
        else:
            k, v = line.split()
            (filt[1] if filt else out[cur])[k] = float(v)
    return out


def apply_chain(img, chain):
    for name, p in chain:
        if name == 'SobelFilter': img = sobel(img, p.get('Scale', 1.0), p.get('Blend', 0.5))
        elif name == 'ContrastFilter': img = contrast(img, p.get('Alpha', 1.0), p.get('Beta', 1.0))
        elif name == 'GaussianFilter': img = gaussian(img, p.get('Radius', 1.0))
        elif name == 'SharpenFilter': img = sharpen(img, p.get('Radius', 1.0), p.get('Contrast', 1.0))
        else: raise ValueError(name)
    return img
