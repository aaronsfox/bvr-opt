# General tracking: options reference

Everything that can be changed when running `general_tracking`, in one place. See `README.md` for what the
pipeline does and what it writes.

There are two ways to run it:

- **On the HPC** with `general_tracking/slurm/submit.sh`. It takes positional arguments plus optional
  environment variables (section 1), and chains keyframe correction → gaps → writing the `.tra` files.
- **Directly** with `python general_tracking/run.py <command> --option ...` (section 2), e.g. to check a trial
  on CPU or to run everything in one process.

Method settings shared by every run are in `general_tracking/config.yaml` (section 3). Run every command from
the repo root (`bvr-opt/`).

## 1. `submit.sh` (HPC)

```bash
[VARIABLE=value ...] bash general_tracking/slurm/submit.sh CFG TRA SPACING [WORKERS] [TIME]
```

### Positional arguments

| Argument | Required | Default | What it sets |
|---|---|---|---|
| `CFG` | yes | – | The trial's Autoscoper `.cfg`, e.g. `datasets/Ballet/003_ShoeRise_R/003_Config.cfg` |
| `TRA` | yes | – | Folder holding the keyframe `.tra` files (one per bone) |
| `SPACING` | yes | – | `N`, `full` or `all`; several separated by colons, e.g. `5:10:20:full` (see section 4) |
| `WORKERS` | no | `4` | Processes sharing the one GPU in the gaps job |
| `TIME` | no | `12:00:00` | SLURM time limit for each GPU job |

`CFG` and `TRA` must not contain commas (SLURM splits `--export` values at commas).

### Optional environment variables

Put them in front of the command, e.g. `BONES=1stMet:1stPhal PER_BONE=1 bash general_tracking/slurm/submit.sh ...`.
Separate multiple values with colons, not commas.

| Variable | Default | Same as | What it does |
|---|---|---|---|
| `BONES` | all volumes in the `.cfg` | `--bones` | Bones to track, e.g. `1stMet:2ndMet:1stPhal`. Other volumes and their `.tra` files are ignored |
| `TRA_PATTERN` | `*.tra` | `--tra-pattern` | Use only `.tra` files matching this, case-insensitive, e.g. `"*_Tracking.tra"` to skip `*_Initial.tra`. Keep the quotes |
| `PER_BONE` | off | `--per-bone` | `PER_BONE=1`: fit each bone alone between its own keyframes (section 5) |
| `KC_TOL` | `1.5:2` | `--kc-tol` | Keyframe-correction limit `mm:deg`. `0:0` keeps keyframes exactly as given and skips the keyframe-correction job |
| `VIE` | `xParameters/control_settings.vie` next to the `.cfg` | `--vie` | Renderer settings file, if not in the default place |
| `D_MIN` | `-2` (config) | `--d-min` | Contact allowance in mm (section 3), e.g. `D_MIN=-1` |
| `MU` | `0.01` (config) | `--mu` | Contact penalty weight, e.g. `MU=0.05` |
| `BVR_RESULTS` | `results/general_tracking/<trial>` | `--out` | Results folder. Use a new one for each changed setting (section 6) |

`<trial>` is the `.cfg` file name without `.cfg`.

## 2. `run.py` (direct)

```bash
python general_tracking/run.py COMMAND --cfg CFG --tra TRA --spacing SPACING [options]
```

### Commands

| Command | What it does | Where to run |
|---|---|---|
| `keys` | Loads the trial and prints each spacing's keyframes and gaps, the keyframe files (with row/column order) and the `.vie` used. Changes nothing | CPU, seconds |
| `run` | Runs keyframe correction, then the gaps, then `assemble`, in one process (or one stage with `--stage`) | GPU (CPU only for short tests) |
| `assemble` | Writes the `optimised/` and `spline/` `.tra` files, `keyframes.json` and `diagnostics.csv` from finished work. Safe to run early: unfinished gaps are NaN | CPU, about a minute |
| `ngaps` | Prints the number of gaps to fill | CPU, seconds |

### Options

| Option | Default | What it does |
|---|---|---|
| `--cfg PATH` | – (required) | Trial `.cfg`. Paths inside it are relative to its folder; Windows separators are fine |
| `--tra PATH` | – (required) | Folder of keyframe `.tra` files, or explicit pairs `Bone=path,Bone=path` |
| `--spacing S` | – (required) | `N`, `full` or `all`; several comma-separated, e.g. `5,10,full` |
| `--bones B` | all volumes | Bones to track, comma- or colon-separated |
| `--tra-pattern P` | `*.tra` | Only `.tra` files matching `P` (case-insensitive) |
| `--per-bone` | off | Fit each bone alone between its own keyframes (section 5) |
| `--kc-tol MM:DEG` | `1.5:2` (config) | Keyframe-correction limit; `0:0` = no correction |
| `--d-min MM` | `-2` (config) | Contact allowance: overlap allowed before the contact penalty starts (section 3). Not used in per-bone runs |
| `--mu W` | `0.01` (config) | Contact penalty weight (≥ 0). Not used in per-bone runs |
| `--vie PATH` | `xParameters/control_settings.vie` next to the `.cfg` | Renderer settings; Welte's defaults if none is found (logged) |
| `--out PATH` | `$BVR_RESULTS`, else `results/general_tracking/<trial>` | Results folder |
| `--stage S` | `all` | `keys`, `gaps` or `all` (submit.sh uses `keys` then `gaps`) |
| `--task I --ntasks N` | `0`, `1` | Share the gaps (and per-bone keyframe correction) between N processes; set by the job scripts |
| `--quick` | off | Smoke test: 2 iterations per stage, nothing saved |

### Environment variables read by `run.py`

| Variable | Default | What it does |
|---|---|---|
| `BVR_DEVICE` | `auto` (GPU if available) | `cuda` or `cpu`. Use `BVR_DEVICE=cpu` for `keys`, `assemble` and smoke tests on a login node |
| `BVR_RESULTS` | – | Results folder when `--out` is not given |
| `THREADS` | `SLURM_CPUS_PER_TASK`, else `2` | CPU threads per process |

## 3. Method settings (`general_tracking/config.yaml`)

These apply to every run from that copy of the repo. `d_min` and `mu` can instead be set per run with
`--d-min` / `--mu` (`D_MIN` / `MU` for `submit.sh`); the values used are recorded in the work files and
`keyframes.json`. Edit the others in a branch or note the values used, and run in a new results folder
(section 6).

| Setting | Current | What it controls |
|---|---|---|
| `d_min` | `-2.0` mm | Contact allowance: how far one bone's surface may sit inside another's before the contact penalty starts. Negative = allowed overlap; `0` = no overlap allowed; positive = required gap. Set for Welte's expanded volumes; tightly segmented joints may suit `-1` to `-0.5` |
| `mu` | `0.01` | Contact penalty weight (gentle). Raise towards `0.05`–`0.1` if bones still overlap too much; too high overrides the images |
| `keyframe_correction.tol` | `[1.5, 2.0]` | Default keyframe-correction limit (mm, deg); `--kc-tol` / `KC_TOL` override it per run |
| `keyframe_correction.window` | `2` | Frames either side of a keyframe used to correct it |
| `w_drr`, `w_f2f` | `1.0`, `2.0` | Weights of the image term and the frame-to-frame term |
| `K_per_10`, `batch_per_10` | `3`, `4` | Smooth-correction terms and sampled frames per 10 frames of gap (stage 3) |
| `res`, `iters`, `lrs` | `[256, 512]`, `[40, 50]`, `[0.3, 0.1]` | Render resolutions, iterations and learning rates of stage 3 (and of keyframe correction) |
| `step_res`, `step_iters`, `step_lrs` | `[256, 512]`, `[20, 20]`, `[0.2, 0.05]` | The same for each frame of the stepwise chains (stage 2) |
| `spline` | `spline` | Interpolation used for the spline output |

The defaults are the settings validated on the Welte benchmark. Change the contact settings for a new
dataset's segmentation if needed; leave the rest unless testing the method itself.

## 4. Keyframe spacing

| Value | Keyframes used |
|---|---|
| `N` | Every N frames from the first frame keyed for all bones (the first and last such frames are always keyframes; a final gap shorter than N/2 joins the one before; a keyframe on an unkeyed frame moves to the nearest keyed one). For testing against full tracking |
| `full` | The first and last keyed frames only |
| `all` | Every keyed frame, as given. **Use this to fill the gaps between your own keyframes** |

Outputs go to `s<spacing>/`, e.g. `s10/`, `sall/` (`s<spacing>_perbone/` in per-bone mode).

## 5. Joint vs per-bone fitting

| | Joint (default) | Per-bone (`PER_BONE=1` / `--per-bone`) |
|---|---|---|
| Keyframes | Only frames keyed for **every** bone | Each bone's own keyframes |
| Bones rendered together | Yes (overlapping bones modelled) | No |
| Contact between bones | Penalised beyond `d_min` | Not checked |
| Results folder | `s<spacing>/` | `s<spacing>_perbone/` (work files in `work/<bone>/`) |

Joint and per-bone runs never overwrite or reuse each other's work.

## 6. Results folders and reruns

- Finished work (`work/`) is reused: resubmitting the same command resumes where it stopped.
- **Changing a setting** (`d_min`, `mu`, `KC_TOL`, bones, the keyframe files, other config values) **needs a
  new results folder**: e.g.
  `D_MIN=-1 BVR_RESULTS=results/general_tracking/003_ShoeRise_R_dmin1 bash general_tracking/slurm/submit.sh ...`.
  For `d_min`, `mu` and `KC_TOL` the run stops with a message if the folder holds work made with other values;
  for the rest it would silently reuse the old work.
- Changing only the spacing or joint/per-bone needs nothing: each has its own `s...` folder.
- Logs: worker logs in `<results folder>/logs/`; SLURM's own `.out` files in
  `results/general_tracking/<trial>/logs/`.

## 7. Input files

| Input | Requirements |
|---|---|
| `.cfg` | Autoscoper Version 1.1 |
| Calibrations | MayaCam 2.0 or 1.0 (detected as Autoscoper does); image size must match the radiographs |
| Radiographs | One `.tif` per frame per camera, 8- or 16-bit, sorted by file name |
| Volumes | One per bone; the bone name is the file name (`1stMet.tif` → `1stMet`; `X_PVtib.tif` → `tib`). `VolumeFlip` is applied (not yet checked against Autoscoper on a flipped dataset) |
| Keyframe `.tra` | One per bone, file name starting with the bone name; one line per frame, `NaN` for untracked frames; row- or column-major (detected per file; outputs written the same way) |

## Examples

```bash
# Check a trial (CPU, seconds)
BVR_DEVICE=cpu python general_tracking/run.py keys --cfg datasets/Ballet/003_ShoeRise_R/003_Config.cfg \
    --tra datasets/Ballet/003_ShoeRise_R/Tracking --bones 1stMet,2ndMet,1stPhal \
    --tra-pattern "*_Tracking.tra" --spacing all

# Joint fit over the frames keyed for all three bones
BONES=1stMet:2ndMet:1stPhal TRA_PATTERN="*_Tracking.tra" bash general_tracking/slurm/submit.sh \
    datasets/Ballet/003_ShoeRise_R/003_Config.cfg datasets/Ballet/003_ShoeRise_R/Tracking all

# Each bone over its own keyframes
BONES=1stMet:2ndMet:1stPhal TRA_PATTERN="*_Tracking.tra" PER_BONE=1 bash general_tracking/slurm/submit.sh \
    datasets/Ballet/003_ShoeRise_R/003_Config.cfg datasets/Ballet/003_ShoeRise_R/Tracking all

# Keyframes held exactly as given, separate results folder
KC_TOL=0:0 BVR_RESULTS=results/general_tracking/003_ShoeRise_R_nokc BONES=1stMet:1stPhal \
    bash general_tracking/slurm/submit.sh datasets/Ballet/003_ShoeRise_R/003_Config.cfg \
    datasets/Ballet/003_ShoeRise_R/Tracking all

# Joint fit with a tighter contact allowance, in its own results folder
D_MIN=-1 BVR_RESULTS=results/general_tracking/003_ShoeRise_R_dmin1 BONES=1stMet:2ndMet:1stPhal \
    TRA_PATTERN="*_Tracking.tra" bash general_tracking/slurm/submit.sh \
    datasets/Ballet/003_ShoeRise_R/003_Config.cfg datasets/Ballet/003_ShoeRise_R/Tracking all

# Write the .tra files from whatever has finished (CPU)
BVR_DEVICE=cpu python general_tracking/run.py assemble --cfg ... --tra ... --spacing all [--per-bone]
```
