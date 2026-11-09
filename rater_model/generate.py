"""Generate simulated rater poses for every frame of each trial, from the calibrated model.

    python rater_model/generate.py          # after calibrate.py

For each rater (n_raters), trial and bone, one error track is drawn over all frames at unit level, and each
rater level multiplies it (levels 1, 2, 4 in config.yaml), so a rater keeps the same error pattern at every
level and every keyframe spacing (paired comparisons). Keyframes for any spacing are then taken from these
tracks. At level 1 (trained), frames beyond Welte's operator standard (helical rotation > 2 deg or helical
translation > 1 mm) have their frame-to-frame part redrawn, since Welte's operators were held to it. Higher
levels are not truncated.

Writes results/rater_model/raters/L<level>/r<id>/<trial>.npz with {bone: (n_frames, 4, 4) rater poses,
bone_rot: (n_frames, 3) rotation-vector error (deg), bone_trans: (n_frames, 3) pivot translation error (mm)},
and results/rater_model/raters/summary.md.
"""
import json
import os
import sys
import zlib

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.io import Trial  # noqa: E402
from src.raters import error_matrix, helical_metrics  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, 'rater_model', 'config.yaml')))
OUT = os.path.join(ROOT, 'results', 'rater_model')
TH_ROT, TH_TRANS = 2.0, 1.0


def seed(*parts):
    return zlib.crc32('|'.join(map(str, parts)).encode())


def unit_track(P, b, n_frames, rng):
    """Unit-level track: offset + AR(1) parts kept separate so frames can be redrawn at level 1."""
    out = {}
    for q, share in (('rotation_deg', P['offset_share'][b]), ('translation_mm', P['offset_share_trans'][b])):
        s = P['scale'][b][q]; sr = np.sqrt(1 - share) * s; phi = P['phi']
        o = rng.normal(0, np.sqrt(share) * s, 3)
        a = np.empty((n_frames, 3)); a[0] = rng.normal(0, sr, 3)
        for f in range(1, n_frames):
            a[f] = phi * a[f - 1] + np.sqrt(1 - phi ** 2) * rng.normal(0, sr, 3)
        out[q] = (o, a, sr)
    return out


def gate(track, rng, max_tries=200):
    """Level 1: redraw the frame-to-frame part of frames beyond 2 deg / 1 mm (helical), keeping the offset."""
    (orr, ar, sr_r), (ot, at, sr_t) = track['rotation_deg'], track['translation_mm']
    redrawn = 0
    for f in range(len(ar)):
        for _ in range(max_tries):
            ang, ax, _ = helical_metrics(orr + ar[f], ot + at[f])
            if ang <= TH_ROT and ax <= TH_TRANS:
                break
            ar[f] = rng.normal(0, sr_r, 3); at[f] = rng.normal(0, sr_t, 3); redrawn += 1
    return redrawn


def main():
    P = json.load(open(os.path.join(OUT, 'params.json')))
    bones = list(CFG['bones'])
    stats = []
    for i in range(CFG['n_raters']):
        for tn in CFG['trials']:
            T = Trial(os.path.join(ROOT, CFG['trials_root'], tn)); N = T.n_frames
            unit = {b: unit_track(P, b, N, np.random.default_rng(seed('rater', i, tn, b))) for b in bones}
            for L in CFG['levels']:
                out = {}
                for b in bones:
                    tr = {q: (o.copy(), a.copy(), sr) for q, (o, a, sr) in unit[b].items()}
                    n_red = gate(tr, np.random.default_rng(seed('gate', i, tn, b))) if L == 1 else 0
                    r = L * (tr['rotation_deg'][0] + tr['rotation_deg'][1])
                    t = L * (tr['translation_mm'][0] + tr['translation_mm'][1])
                    piv = T.bones[b].pivot
                    out[b] = np.stack([T.ref[b][f] @ error_matrix(r[f], t[f], piv) for f in range(N)])
                    out[f'{b}_rot'] = r; out[f'{b}_trans'] = t
                    ang, ax, pv = helical_metrics(r, t)
                    stats.append(dict(level=L, rater=i, trial=tn, bone=b, rot=ang, ax=ax, piv=pv, redrawn=n_red, n=N))
                path = os.path.join(OUT, 'raters', f'L{L}', f'r{i:02d}', f'{tn}.npz'); os.makedirs(os.path.dirname(path), exist_ok=True)
                np.savez(path, **out)
    # summary
    lines = ['# Simulated raters', '', f"{CFG['n_raters']} raters × {len(CFG['trials'])} trials × levels {CFG['levels']}. "
             'Per-frame errors pooled over raters, trials and frames: median [95th percentile]. Helical metrics as in '
             'Welte 2022; pivot displacement is the full positional error of the bone pivot.', '',
             '| Level | Bone | Helical rotation ° | Helical translation mm | Pivot displacement mm | Frames beyond 1 mm / 2° | Frames redrawn (level 1) |',
             '|---|---|---|---|---|---|---|']
    for L in CFG['levels']:
        for b in bones:
            g = [s for s in stats if s['level'] == L and s['bone'] == b]
            rot = np.concatenate([s['rot'] for s in g]); ax = np.concatenate([s['ax'] for s in g]); pv = np.concatenate([s['piv'] for s in g])
            n = sum(s['n'] for s in g)
            lines.append(f"| {L}× | {CFG['bones'][b]} | {np.median(rot):.2f} [{np.percentile(rot, 95):.2f}] | "
                         f"{np.median(ax):.2f} [{np.percentile(ax, 95):.2f}] | {np.median(pv):.2f} [{np.percentile(pv, 95):.2f}] | "
                         f"{100 * np.mean((rot > TH_ROT) | (ax > TH_TRANS)):.1f}% | {100 * sum(s['redrawn'] for s in g) / n:.1f}% |")
    s = '\n'.join(lines) + '\n'; open(os.path.join(OUT, 'raters', 'summary.md'), 'w').write(s); print(s)


if __name__ == '__main__':
    main()
