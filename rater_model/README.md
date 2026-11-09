# Rater model

Simulated rater keyframes, calibrated to Welte et al. (2022) Figures 4 and 5 (digitised values in `welte2022/`).

```bash
python rater_model/calibrate.py     # fit and check -> results/rater_model/params.json, calibration.md, figures/calibration.png
python rater_model/generate.py      # rater poses for every frame -> results/rater_model/raters/L<level>/r<id>/<trial>.npz
```

**Model** (`src/raters.py`). For each rater, trial and bone, every frame's pose is the reference pose with a
rigid error applied in the bone's CT frame about its pivot: a rotation vector and a pivot translation. Each
is a constant offset for the trial (the rater's bias, random size and direction) plus a frame-to-frame part
that drifts smoothly (AR(1)). Directions are isotropic and rotation and translation independent.

**Calibration** (`calibrate.py`). Welte measured error with the helical axis: the rotation angle and the
translation along that axis (not the full displacement). The calibration simulates many raters over a 41-frame
trial, scores them the same way and fits:
- the AR(1) coefficient to Figure 5's frame-to-frame correlation;
- the rotation offset share from Figure 5 (calcaneus and tibia 75 %, talus none);
- one translation offset share for all bones, jointly to Figure 4's translation ranges and the spread of the
  three operators' medians;
- the error scale per bone and quantity to the operators' mean median (exact).

Figure 4's "vertical lines" as digitised are the range of the 41 frames, not the interquartile range (they
match the minimum and maximum of Figure 5's frames). `calibration.md` lists every fitted and checked value.

**Raters** (`generate.py`). 10 raters per level; levels 1× (trained, gated to Welte's 1 mm / 2° helical
standard as their operators were), 2× and 4×. Each rater's error track is drawn once and scaled for each
level, so comparisons across levels and keyframe spacings are paired. Keyframes for any spacing are taken
from these tracks.
