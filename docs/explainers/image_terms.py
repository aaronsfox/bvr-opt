"""Visual explainer: what the image refinement's image and frame-to-frame terms see and compare.

    python docs/explainers/image_terms.py [--trial T0014_hop108_0001] [--frame 35] [--cam 0]

Uses the same functions as the tracking (src.costs.drr_term / gradcorr / F2F, src.filters.blur / sobel_xy,
src.drr.View) on a Welte SOL001A frame at the bead-based reference poses, at the 512 px render grid.
Writes docs/explainers/fig1_processing.png, fig2_comparison.png, fig3_landscape.png, fig4_frame_to_frame.png.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src import plotstyle as ps  # noqa: E402
from src.costs import F2F, fov_masks, gradcorr, sample_mask  # noqa: E402
from src.tracking import drr_term  # noqa: E402
from src.drr import View  # noqa: E402
from src.filters import blur, sobel_xy  # noqa: E402
from src.io import Trial  # noqa: E402
from src.raters import error_matrix  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
BONES = ['tib', 'tal', 'cal']
NAME = {'tib': 'Tibia', 'tal': 'Talus', 'cal': 'Calcaneus'}
RES = 512


def rt(M):
    return torch.tensor(M[:3, :3]), torch.tensor(M[:3, 3])


def perturb(T, n, M, rot=(0, 0, 0), trans=(0, 0, 0)):
    return M @ error_matrix(rot, trans, T.bones[n].pivot)


def render(T, v, poses, fov):
    """Everything the image term computes for one camera: joint DRR, radiograph, both blurred, masks."""
    rois = {n: v.roi(T.bones[n], *rt(poses[n])) for n in BONES}
    lo, hi = v.union_roi(list(rois.values())); rays = v._rays(lo, hi)
    parts = v.render_parts([(T.bones[n], *rt(poses[n])) for n in BONES], lo, hi, rays)
    joint_raw = torch.clamp(sum(parts), 0, 1)
    fm = sample_mask(v, fov, lo, hi).double()
    masks = {n: F.max_pool2d((p > 0).float()[None, None], 7, 1, 3)[0, 0].double() * fm for n, p in zip(BONES, parts)}
    return lo, hi, joint_raw, masks


def rad_crop(T, v, cam, frame, lo, hi):
    return 1 - v.sample_image(T.image(cam, frame, RES), lo, hi).double()


def show(ax, im, title, cmap='gray', sym=False, extent=None):
    im = im.detach().cpu().numpy() if torch.is_tensor(im) else im
    if sym:
        s = np.percentile(np.abs(im), 99.5) or 1
        ax.imshow(im, cmap='RdBu_r', vmin=-s, vmax=s, extent=extent)
    else:
        ax.imshow(im, cmap=cmap, extent=extent)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def fig_processing(T, v, cam, frame, poses, fov):
    lo, hi, drr_raw, masks = render(T, v, poses, fov)
    rad_raw = rad_crop(T, v, cam, frame, lo, hi)
    rad_b, drr_b = blur(rad_raw, 1.0), blur(drr_raw, 1.0)
    rx, ry = sobel_xy(rad_b); dx, dy = sobel_xy(drr_b)
    full = T.raw_image(cam, frame)
    fig, ax = plt.subplots(2, 5, figsize=(ps.DOUBLE, 3.6))
    show(ax[0, 0], full, f'Camera {cam + 1}, frame {frame}')
    c0, r0 = lo; c1, r1 = hi
    ax[0, 0].plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0], color=ps.color(1), lw=1, marker='')
    show(ax[0, 1], rad_raw, 'Inverted, 512 px grid')
    show(ax[0, 2], rad_b, 'Blurred, σ = 1 px')
    show(ax[0, 3], rx, 'Sobel x (signed)', sym=True)
    show(ax[0, 4], ry, 'Sobel y (signed)', sym=True)
    m = sum(masks.values()).clamp(0, 1).cpu().numpy()
    show(ax[1, 0], m, 'Scoring masks')
    show(ax[1, 1], drr_raw, 'DRR, 3 bones')
    show(ax[1, 2], drr_b, 'Blurred, σ = 1 px')
    show(ax[1, 3], dx, 'Sobel x (signed)', sym=True)
    show(ax[1, 4], dy, 'Sobel y (signed)', sym=True)
    ax[0, 0].text(-0.08, 0.5, 'Radiograph', rotation=90, transform=ax[0, 0].transAxes, va='center', ha='right', fontsize=9, fontweight='bold')
    ax[1, 0].text(-0.08, 0.5, 'DRR', rotation=90, transform=ax[1, 0].transAxes, va='center', ha='right', fontsize=9, fontweight='bold')
    fig.tight_layout(); ps.save(fig, os.path.join(OUT, 'fig1_processing.png'), pdf=False); plt.close(fig)


def agreement(T, v, cam, frame, poses, fov, bone):
    """Per-pixel agreement of DRR and radiograph edges inside one bone's mask, and that bone's 1 - GC."""
    lo, hi, drr_raw, masks = render(T, v, poses, fov)
    rad_b, drr_b = blur(rad_crop(T, v, cam, frame, lo, hi), 1.0), blur(drr_raw, 1.0)
    rx, ry = sobel_xy(rad_b); dx, dy = sobel_xy(drr_b)
    m = masks[bone]
    prod = ((rx - rx[m > 0].mean()) * (dx - dx[m > 0].mean()) + (ry - ry[m > 0].mean()) * (dy - dy[m > 0].mean())) * m
    cost = float(1 - gradcorr(drr_b, rad_b, m))
    mag_r = torch.sqrt(rx ** 2 + ry ** 2); mag_d = torch.sqrt(dx ** 2 + dy ** 2)
    allm = sum(masks.values()) > 0
    mag_r = mag_r / torch.quantile(mag_r[allm], 0.98); mag_d = mag_d / torch.quantile(mag_d[allm], 0.98)
    return mag_r, mag_d, prod, m, cost


def fig_comparison(T, v, cam, frame, poses, fov, bone='tal'):
    wrong = dict(poses); wrong[bone] = perturb(T, bone, poses[bone], rot=(0, 0, 3), trans=(2, 0, 0))
    fig, ax = plt.subplots(2, 3, figsize=(ps.DOUBLE * 0.85, 4.4))
    for i, (pp, lab) in enumerate(((poses, 'Reference pose'), (wrong, f'{NAME[bone]} moved 2 mm and 3°'))):
        mag_r, mag_d, prod, m, cost = agreement(T, v, cam, frame, pp, fov, bone)
        rgb = np.zeros(mag_r.shape + (3,))
        rgb[..., 0] = mag_d.clamp(0, 1).cpu().numpy()                 # DRR edges: red
        rgb[..., 1] = mag_r.clamp(0, 1).cpu().numpy()                 # radiograph edges: green
        rgb[..., 2] = rgb[..., 1] * 0.3
        show(ax[i, 0], np.clip(rgb, 0, 1), f'{lab}: edges\n(green radiograph, red DRR, yellow = both)')
        show(ax[i, 1], m, f'{NAME[bone]} scoring mask')
        s = float(np.percentile(np.abs(prod.cpu().numpy()), 99.5)) or 1
        ax[i, 2].imshow(prod.cpu().numpy(), cmap='RdBu_r', vmin=-s, vmax=s); ax[i, 2].set_xticks([]); ax[i, 2].set_yticks([])
        ax[i, 2].set_title(f'Edge agreement in mask\n(red agree, blue oppose); 1 − GC = {cost:.3f}', fontsize=8)
        for s_ in ax[i, 2].spines.values():
            s_.set_visible(False)
    fig.tight_layout(); ps.save(fig, os.path.join(OUT, 'fig2_comparison.png'), pdf=False); plt.close(fig)


def fig_landscape(T, frame, poses, fovs, bone='tal'):
    """Image term (both cameras, the moved bone's terms only) as one bone is moved off its reference pose."""
    ps.apply()
    rads = {r: [T.image(ci, frame, r) for ci in range(len(T.cams))] for r in (256, 512)}
    views = {r: [View(c, r, T.sample_distance, T.ray_intensity) for c in T.cams] for r in (256, 512)}

    def cost(res, rot=(0, 0, 0), trans=(0, 0, 0)):
        pp = dict(poses); pp[bone] = perturb(T, bone, poses[bone], rot, trans)
        with torch.no_grad():
            tot, _ = drr_term(views[res], rads[res], fovs, {bone: T.bones[bone]}, {bone: rt(pp[bone])})
        return float(tot)

    tr = np.linspace(-4, 4, 33); ro = np.linspace(-8, 8, 33)
    fig, ax = plt.subplots(1, 2, figsize=(ps.DOUBLE, 2.6), sharey=True)
    for res, ls in ((512, '-'), (256, '--')):
        for k, axis in enumerate('xyz'):
            st = dict(ps.style(k, hollow=res == 256), ms=3.5, lw=1.3, markevery=2)
            ax[0].plot(tr, [cost(res, trans=tuple(t if j == k else 0 for j in range(3))) for t in tr], ls,
                       label=f'{axis}, {res} px', **st)
            ax[1].plot(ro, [cost(res, rot=tuple(a if j == k else 0 for j in range(3))) for a in ro], ls,
                       label=f'about {axis}, {res} px', **st)
    ax[0].set_xlabel('Translation of the pivot (mm)'); ax[1].set_xlabel('Rotation about the pivot (°)')
    ax[0].set_ylabel(f'Image term, {NAME[bone]} (sum of 1 − GC, 2 views)')
    for a in ax:
        a.axvline(0, color='0.6', lw=0.8, ls=':')
    ax[0].legend(fontsize=6, ncol=2); ax[1].legend(fontsize=6, ncol=2)
    ax[0].set_title(f'{NAME[bone]} moved along each CT axis', fontsize=9); ax[1].set_title(f'{NAME[bone]} rotated about each CT axis', fontsize=9)
    fig.tight_layout(); ps.save(fig, os.path.join(OUT, 'fig3_landscape.png'), pdf=False); plt.close(fig)


def fig_f2f(T, cam, frame, poses_f, poses_g, fovs, bone='tal'):
    """Edge points of one bone placed with each frame's pose and drawn on that frame's radiograph gradients."""
    f2f = F2F(T, [frame, frame + 1], fovs)
    P = T.bones[bone].surface_points(2500, 0.6, seed=0)
    pts = lambda M: torch.tensor(P @ M[:3, :3].T + M[:3, 3])
    wrong = perturb(T, bone, poses_g[bone], rot=(0, 0, 3), trans=(2, 0, 0))
    fig, ax = plt.subplots(1, 3, figsize=(ps.DOUBLE, 2.8))
    c0, r0, c1, r1 = None, None, None, None
    sets = [(frame, poses_f[bone], f'Frame {frame}: points at its pose'),
            (frame + 1, poses_g[bone], f'Frame {frame + 1}: points at its pose'),
            (frame + 1, wrong, f'Frame {frame + 1}: pose 2 mm, 3° off')]
    uvs = [np.stack([t.numpy() for t in f2f.project(cam, pts(M))], 1) for _, M, _ in sets]
    allp = np.vstack(uvs); c0, r0 = allp.min(0) - 25; c1, r1 = allp.max(0) + 25
    vals = []
    for (f, M, lab), uv, a in zip(sets, uvs, ax):
        G = f2f.G[(f, cam)][0].cpu().numpy(); mag = np.hypot(G[0], G[1])
        a.imshow(mag, cmap='gray', vmax=np.percentile(mag, 99.5)); a.set_xlim(c0, c1); a.set_ylim(r1, r0)
        v, _ = f2f.sample(f, cam, *f2f.project(cam, pts(M))); vals.append(v)
        a.scatter(uv[::4, 0] - 0.5, uv[::4, 1] - 0.5, s=1.5, c=v[0].numpy()[::4], cmap='RdBu_r', linewidths=0)
        a.set_title(lab, fontsize=8); a.set_xticks([]); a.set_yticks([])
    ncc = lambda x, y: float(F2F._ncc(x, y))
    good = 0.5 * (ncc(vals[0][0], vals[1][0]) + ncc(vals[0][1], vals[1][1]))
    bad = 0.5 * (ncc(vals[0][0], vals[2][0]) + ncc(vals[0][1], vals[2][1]))
    ax[1].set_xlabel(f'agreement with frame {frame}: NCC = {good:.2f}', fontsize=8)
    ax[2].set_xlabel(f'agreement with frame {frame}: NCC = {bad:.2f}', fontsize=8)
    fig.suptitle(f'{NAME[bone]} edge points (every 4th of 2,500) on radiograph edge strength, camera {cam + 1}; '
                 'colour = sampled x-gradient', fontsize=8)
    fig.tight_layout(); ps.save(fig, os.path.join(OUT, 'fig4_frame_to_frame.png'), pdf=False); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trial', default='T0014_hop108_0001'); ap.add_argument('--frame', type=int, default=35)
    ap.add_argument('--cam', type=int, default=0)
    a = ap.parse_args()
    ps.apply()
    T = Trial(os.path.join(ROOT, 'datasets', 'Welte2022', 'data', 'SOL001A', a.trial))
    fovs = fov_masks(T)
    v = View(T.cams[a.cam], RES, T.sample_distance, T.ray_intensity)
    poses = {n: T.ref[n][a.frame] for n in BONES}; nxt = {n: T.ref[n][a.frame + 1] for n in BONES}
    with torch.no_grad():
        fig_processing(T, v, a.cam, a.frame, poses, fovs[a.cam])
        fig_comparison(T, v, a.cam, a.frame, poses, fovs[a.cam])
        fig_f2f(T, a.cam, a.frame, poses, nxt, fovs)
    fig_landscape(T, a.frame, poses, fovs)
    print('written to', OUT)


if __name__ == '__main__':
    main()
