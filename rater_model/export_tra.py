"""Export the simulated raters' keyframes as Autoscoper .tra files, one set per keyframe spacing.

    python rater_model/export_tra.py          # after generate.py

For each level, rater and keyframe spacing (the layouts used by keyframe_gaps: bookends at frame 5 and
N - 6, keyframes every s frames, `full` = bookends only), writes one .tra per trial and bone with the rater's
pose on keyframes and NaN on every other frame. The reference (bead-based) keyframes are exported the same
way under `reference/`, for comparison. Files use Welte's naming and layout (<trial>_<bone>.tra,
column-major 4x4 per line), so they load into Autoscoper with the same import settings as the reference
files: Matrix, Column, Comma, mm, degrees.

Writes results/rater_model/tra/{L<level>/r<id>,reference}/s<spacing>/<trial>_<bone>.tra and tra/README.md.
"""
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'keyframe_gaps')); sys.path.insert(1, ROOT)
import run as kg  # noqa: E402  (keyframe layouts)
from src.io import write_tra  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(ROOT, 'rater_model', 'config.yaml')))
SRC = os.path.join(ROOT, 'results', 'rater_model', 'raters')
OUT = os.path.join(ROOT, 'results', 'rater_model', 'tra')
BONES = list(CFG['bones']); SPACINGS = [str(s) for s in kg.CFG['spacings']]


def keyframes_only(M, keys):
    out = np.full_like(M, np.nan)
    out[keys] = M[keys]
    return out


def export(poses, n_frames, folder, trial):
    """poses: {bone: (n_frames, 4, 4)} -> one keyframe-only .tra per bone and spacing under folder/s<spacing>/."""
    for s in SPACINGS:
        keys = kg.keyframes(n_frames, s); d = os.path.join(folder, f's{s}'); os.makedirs(d, exist_ok=True)
        for b in BONES:
            write_tra(os.path.join(d, f'{trial}_{b}.tra'), keyframes_only(poses[b], keys))


def main():
    n = 0
    for tn in CFG['trials']:
        T = kg.trial(tn)
        export({b: T.ref[b] for b in BONES}, T.n_frames, os.path.join(OUT, 'reference'), tn); n += 1
        for L in CFG['levels']:
            for i in range(CFG['n_raters']):
                d = np.load(os.path.join(SRC, f'L{L}', f'r{i:02d}', f'{tn}.npz'))
                export({b: d[b] for b in BONES}, T.n_frames, os.path.join(OUT, f'L{L}', f'r{i:02d}'), tn); n += 1
    layout = '\n'.join(f'| {kg.trial(tn).n_frames} | {tn} | ' + ' | '.join(
        ','.join(map(str, kg.keyframes(kg.trial(tn).n_frames, s))) for s in SPACINGS) + ' |' for tn in CFG['trials'])
    open(os.path.join(OUT, 'README.md'), 'w').write(f"""# Rater keyframes as Autoscoper .tra files

Written by `rater_model/export_tra.py` from the simulated raters in `../raters/` (the `.npz` files hold every
frame plus the error vectors; these hold keyframes only).

```
tra/
  reference/s<spacing>/<trial>_<bone>.tra        bead-based keyframes (perfect rater)
  L<level>/r<id>/s<spacing>/<trial>_<bone>.tra   simulated rater <id> at level <level> (1 = trained, 2x, 4x)
```

Bones: `cal` calcaneus, `tal` talus, `tib` tibia. Each file has one line per frame of the trial: the 4x4 pose,
column-major as in Welte's reference files (translation in elements 13-15), on keyframes, and NaN on every
other frame. Load in Autoscoper with: Matrix, Orientation Column, Comma, mm, degrees, one bone at a time
(Volume = Current).

Keyframes per spacing (bookends at frame 5 and N - 6; regular keyframes every s frames; `full` = bookends):

| Frames | Trial | {' | '.join('s' + s for s in SPACINGS)} |
|---|---|{'---|' * len(SPACINGS)}
{layout}
""")
    print(f'exported {n} trial sets x {len(SPACINGS)} spacings x {len(BONES)} bones to {OUT}')


if __name__ == '__main__':
    main()
