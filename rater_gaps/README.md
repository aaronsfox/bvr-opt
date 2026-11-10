# Keyframe gaps with simulated raters

Analysis 2 combined with Analysis 3: the stepwise keyframe-correction method (`new_opt_stepwise_kc`) on
keyframes from the simulated raters (`rater_model/`), at keyframe spacings 5, 10, 20, 30, 40 and full trial,
compared with the same spacings on ideal (bead-based) keyframes from `results/keyframe_gaps/`.

- **Raters:** levels 1× (trained, Welte-like), 2× and 4×; 10 raters each (`results/rater_model/raters`).
  Every rater's keyframes at every spacing come from that rater's error track, so levels and spacings are
  paired comparisons.
- **Method:** every rater keyframe is first corrected against the images of the frames around it (±2),
  moving at most the level's tolerance (`kc_tol` in `config.yaml`: 1.5 mm / 2° at 1×, 3 mm / 3° at 2×,
  6 mm / 6° at 4×, about the 95th percentile of each level's keyframe error). The corrected keyframes then
  anchor the stepwise chains and the anchored refinement, as with ideal keyframes.
  The correction takes fixed-size steps (`step: mm` in `keyframe_gaps/config.yaml`): the tolerance only caps
  how far a keyframe can move, not how fast. With steps scaled by the tolerance, the wide 4× tolerance made
  each step several mm and the fit overshot; see `results/rater_gaps/tests/kc_step/summary.md`.
- **Reference:** spline interpolation through the same (uncorrected) rater keyframes.
- **Outcomes:** as keyframe_gaps, plus Welte's helical translation, with frames needing correction counted
  under Welte's criterion (helical rotation > 2° or helical translation > 1 mm) and the positional criterion
  (pivot > 1 mm or rotation > 2°); keyframe error as given by the rater and after correction; spread between
  raters.

## Running on the HPC

Each array task is one rater at one level and spacing, on one `dgx` GPU, for one stage. The keyframe-correction
task (keys) runs first; that rater's gaps task starts as soon as it succeeds. Tasks run from minutes to
about an hour, so they fit into gaps in the queue.

```bash
git pull origin main
bash rater_gaps/slurm/submit_all.sh           # 3 levels x 6 spacings x 10 raters x 2 stages = 360 tasks
# or a piece at a time:  bash rater_gaps/slurm/submit.sh LEVEL SPACING [RATERS] [TIME] [MAX_CONCURRENT]
#   e.g. bash rater_gaps/slurm/submit.sh 1 20          (level 1, spacing 20, raters 0-9)
#        bash rater_gaps/slurm/submit.sh 2 full 0-4    (level 2, full trial, raters 0-4)
# optional MAX_CONCURRENT (e.g. submit_all.sh 2) caps how many of each array's tasks run at once

squeue -u $USER
ls results/rater_gaps/L1/r00/new_opt_stepwise_kc/s20/units | wc -l      # progress for one rater/spacing

sbatch rater_gaps/slurm/score.sbatch          # interpolation reference + assemble + score (CPU; fine part-way)
```

Units per rater: keys stage 4 (one per trial); gaps 36 / 18 / 9 / 6 / 5 / 4 for spacings 5 / 10 / 20 / 30 /
40 / full. Finished units are skipped, so resubmitting a command only redoes what is missing.

## Results

```
results/rater_gaps/
  L<level>/r<id>/new_opt_stepwise_kc/s<spacing>/keyframes|units|tracks|tra
  L<level>/r<id>/new_opt_stepwise_kc_chains/s<spacing>/tracks
  L<level>/r<id>/keyframe_interp/s<spacing>/units|tracks|tra
  scores.csv  keyframes.csv  summary.md  figures/raters_by_spacing.png
  logs/   (not committed)
```

`tracks/<trial>.npz` holds {bone: (n_frames, 4, 4)}; `tra/<trial>_<bone>.tra` holds the same tracks as Autoscoper
files (column-major like Welte's reference files, `NaN` outside the bookends). Load in Autoscoper with Matrix,
Column, Comma, mm, degrees. The rater keyframes themselves are in `results/rater_model/tra/`.
