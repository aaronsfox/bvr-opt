# General tracking

Tracks any Autoscoper trial from keyframes. You give it the trial's `.cfg`, the keyframe tracking (`.tra`, one
per bone) and a keyframe spacing. It returns two sets of tracking as Autoscoper `.tra` files:

- **optimised:** keyframe-anchored image refinement, the paper method (`new_opt_stepwise_kc`), with the Welte
  benchmark settings;
- **spline:** spline interpolation through the same keyframes.

Nothing is scored here. To compare the outputs with other tracking (e.g. a rater's full tracking), use a
separate step on the `.tra` files.

## Inputs

```
<trial folder>/
    <name>.cfg              Autoscoper Version 1.1; paths relative to this folder (Windows separators fine)
    <calibrations>          MayaCam 2.0, one per camera
    <CameraRootDir>/*.tif   one image per frame per camera, sorted by file name; 8- or 16-bit
    <VolumeFile>.tif        one segmented CT volume per bone; the bone is named after the file
                            (Femur.tif -> Femur); VolumeFlip is applied
    xParameters/control_settings.vie   renderer settings, as Autoscoper looks for them (Welte defaults if
                                       absent; each job log names the file used)
<keyframe folder>/*.tra     one per bone, named after it (Femur_keys.tra), as Autoscoper writes them:
                            one line per frame, NaN on frames without a keyframe
```

**Keyframes** are taken only from frames tracked for every bone. The spacing sets which of them are used:

| `--spacing` | Keyframes |
|---|---|
| `N` | every N frames from the first tracked frame; the first and last tracked frames are always keyframes, and a final gap shorter than N/2 joins the one before. A keyframe that lands on an untracked frame moves to the nearest tracked one |
| `full` | the first and last tracked frames only |
| `all` | every tracked frame, i.e. the keyframes exactly as given (use this for sparse keyframes) |

## Running

From the repo root, with the dataset under `datasets/UNC/` (git-ignored):

```bash
C=datasets/UNC/Sub05/Sub05_Walk1550.cfg; K=datasets/UNC/Sub05/Test_Tracking

# Which keyframes each spacing uses (seconds, CPU)
BVR_DEVICE=cpu python general_tracking/run.py keys --cfg $C --tra $K --spacing 5,10,20,full,all

# HPC: keyframe correction -> gaps (4 workers on one GPU) -> .tra files written (CPU)
bash general_tracking/slurm/submit.sh $C $K 10              # one spacing
bash general_tracking/slurm/submit.sh $C $K 5:10:20:full    # several (colon-separated)
#   options as environment variables: BONES=Femur:Tibia  KC_TOL=1.5:2  VIE=file.vie
#   resubmitting the same command resumes; finished work is skipped

# Or in one process (GPU node, or CPU for a short trial)
python general_tracking/run.py run --cfg $C --tra $K --spacing 10

# Smoke test: 2 iterations per stage, nothing saved
BVR_DEVICE=cpu python general_tracking/run.py run --cfg $C --tra $K --spacing full --quick
```

Options: `--kc-tol mm:deg` sets how far keyframe correction may move a keyframe (default 1.5:2, the 95th
percentile of a trained rater's error on Welte; `0:0` holds the keyframes exactly as given). `--bones` selects
a subset of the volumes, `--vie` gives the renderer settings and `--out` sets the results folder.

## Outputs

In `results/general_tracking/<trial>/s<spacing>/` (git-ignored, as it may hold participant data):

```
optimised/<trial>_<bone>.tra   optimised tracking, first to last keyframe (NaN elsewhere); keyframes as corrected
spline/<trial>_<bone>.tra      spline tracking through the keyframes as given
keyframes.json                 keyframe frames, and how far correction moved each keyframe (mm, deg)
diagnostics.csv                per frame and bone: disagreement between the forward and backward chains
                               (mm, deg). Large values flag frames to review (no reference needed)
work/                          intermediate results: corrected keyframes, one file per gap
logs/                          HPC logs
```

Load the `.tra` files into Autoscoper to review every frame.

## Notes

- Run time on Welte: about 32 s per tracked frame with 4 runs sharing an A100.
- The method was developed on Welte's trials, where bones move about 1.5 mm per frame. Larger steps per frame
  make long gaps harder, so keep keyframes closer together for fast motion.
- If the segmented bones routinely overlap by more than 2 mm at the keyframe poses, make `d_min` in
  `config.yaml` more negative.
- VolumeFlip has not yet been checked against Autoscoper on a flipped dataset. Check the first outputs in
  Autoscoper.
