# Keyframe gaps with perfect keyframes

Analysis 3 in the plan doc. How does accuracy change as keyframes get further apart? Keyframes are the
bead-based reference poses (a perfect rater), so the only thing that changes is the spacing.

## Design

- **Trials and bones:** all four SOL001A trials; tibia, talus and calcaneus.
- **Bookends:** every condition has keyframes at frame 5 and frame N − 6. The regular keyframes fall
  every *s* frames from frame 5. If the last regular keyframe would leave a final gap shorter than s/2,
  it is dropped and that gap merges with the one before. `full` uses only the two bookends: one gap over
  the whole trial. `python keyframe_gaps/run.py keys` prints every layout.
- **Spacings:** 5, 10, 20, 30, 40 and full.
- **Short trials:** in the jog (39 frames) and assessment (41 frames) trials, spacings of 20 or more
  collapse to one or two gaps, sometimes the same layout as `full`. The summary lists the gap lengths
  actually used for each trial, and errors are also reported by distance to the nearest keyframe.
- **Scored frames:** every non-keyframe strictly between the bookends. A second table scores every
  spacing on the identical frame set (frames that are a keyframe at no spacing).
- **Outcomes:**
  - per bone: median and 95th percentile of pivot error and helical rotation, plus the maximum;
  - talus–calcaneus relative rotation;
  - the share of bone-frames missing Welte's operator standard (pivot > 1 mm or rotation > 2°), i.e.
    frames needing correction;
  - error by distance to the nearest keyframe;
  - compute.
- **Filtering:** results are unfiltered.

| Arm | Method | Units |
|---|---|---|
| **new_opt** | The methods-comparison method: keyframe-anchored gap fit, bones co-fitted, signed gradient correlation + frame-to-frame + collision (PV, d_min −2 mm), hard anchors. The basis size and frames sampled per iteration scale with gap length (per 10 frames: K = 3, batch = 4), so every frame gets the same flexibility and image data whatever the spacing; iterations are fixed (40 + 50) | one per gap: 36 / 18 / 9 / 6 / 5 / 4 for spacings 5 / 10 / 20 / 30 / 40 / full |
| keyframe_interp | Reference: spline interpolation through the same keyframes (CPU, seconds) | one per trial |

Other methods (e.g. pso_simplex) can be added as arms in `config.yaml`.

## Running on the HPC

From the repo root (uses the `bvr_opt_gpu` env from the methods comparison):

```bash
git pull origin main
bash keyframe_gaps/slurm/submit_all.sh new_opt        # six small job arrays, one per spacing
# or one spacing at a time:  bash keyframe_gaps/slurm/submit.sh new_opt 20

squeue -u $USER
for s in 5 10 20 30 40 full; do echo s$s $(ls results/keyframe_gaps/new_opt/s$s/units 2>/dev/null | wc -l); done

sbatch keyframe_gaps/slurm/score.sbatch               # interpolation reference + assemble + score (CPU)
```

Each job uses one `dgx` GPU shared by 4 worker processes. If any worker fails, the job ends as FAILED,
and the error is in `results/keyframe_gaps/logs/<arm>_s<spacing>_<job>_<task>.log`.

## Results

```
results/keyframe_gaps/
  new_opt/s5/ ... new_opt/sfull/      units/<trial>_<a>-<b>.json, tracks/<trial>.npz
  keyframe_interp/s5/ ...
  scores.csv  compute.csv  summary.md  figures/error_by_spacing.png
  logs/                               (not committed)
```
