"""Contact (interpenetration) penalty between co-fitted bones, after Ito et al. (2015).

Each bone's segmented partial volume (non-zero voxels, holes filled) gives a signed distance field in
its own local frame (mm, positive outside). Surface points of every other co-fitted bone are mapped into
that frame and the field is sampled there (trilinear, differentiable in both poses).
Penalty for a pair = sum over points of max(0, d_min - d)^2, both directions.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .device import dev
from scipy import ndimage as ndi


class SDF:
    def __init__(self, bone, n_surface=3000, seed=0):
        v = bone.vol[0, 0].cpu().numpy()
        m = ndi.binary_fill_holes(v > 0)
        sp = (bone.sz, bone.sy, bone.sx)
        sdf = ndi.distance_transform_edt(~m, sampling=sp) - ndi.distance_transform_edt(m, sampling=sp)
        self.bone = bone
        self.grid = dev(torch.from_numpy(sdf.astype(np.float32)))[None, None]
        surf = m & ~ndi.binary_erosion(m)
        k, r, c = np.nonzero(surf)
        sel = np.random.default_rng(seed).choice(len(k), size=min(n_surface, len(k)), replace=False)
        b = bone
        self.surface = torch.tensor(np.c_[(c[sel] + b.c0 + .5) * b.sx, (b.H0 - r[sel] - b.r0 - .5) * b.sy,
                                          -(k[sel] + b.k0 + .5) * b.sz], dtype=torch.float64)

    def at(self, P_local):
        b = self.bone
        col = P_local[:, 0] / b.sx - b.c0; row = b.H0 - P_local[:, 1] / b.sy - b.r0; sl = -P_local[:, 2] / b.sz - b.k0
        g = torch.stack([2 * col / b.W - 1, 2 * row / b.H - 1, 2 * sl / b.D - 1], -1).float()
        # outside the crop: border padding returns the edge value (positive: outside the bone)
        return F.grid_sample(self.grid, g.view(1, 1, 1, -1, 3), mode='bilinear', padding_mode='border',
                             align_corners=False).view(-1).double()


def pair_distances(A, RA, tA, B, RB, tB):
    """Signed distances (mm) of A's surface points to B and of B's to A."""
    dAB = B.at(((A.surface @ RA.T + tA) - tB) @ RB)
    dBA = A.at(((B.surface @ RB.T + tB) - tA) @ RA)
    return dAB, dBA


def penalty(sdfs, poses, d_min=0.0):
    """sdfs: {name: SDF}; poses: {name: (R, t)}. Sum over pairs of the squared penetration (mm^2)."""
    names = list(poses); tot = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            dAB, dBA = pair_distances(sdfs[a], *poses[a], sdfs[b], *poses[b])
            tot = tot + (torch.relu(d_min - dAB) ** 2).sum() + (torch.relu(d_min - dBA) ** 2).sum()
    return tot


def min_separation(sdfs, poses, a, b):
    with torch.no_grad():
        dAB, dBA = pair_distances(sdfs[a], *poses[a], sdfs[b], *poses[b])
    return float(min(dAB.min(), dBA.min()))
