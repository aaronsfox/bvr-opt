# Ballet pilot code (archived, not maintained)

Exploratory code from the first pilot on the ballet trial (003_Shoe_Rise, frames 146–181,
bones 1stMet, 2ndMet, 1stPhal, 2ndPhal). Kept as a reference for the Welte benchmark implementation.

**Not runnable as-is.** Scripts have hard-coded paths to the original chat session's
uploads (`/mnt/user-data/attachments/...`) and work folder (`/root/work`), and chain together
with `exec(open(...).read())`. No participant data (images, volumes, tracking, FOV masks or
figures showing radiographs) is included.

## Core modules (code/)
| File | Contents |
|---|---|
| `ascam.py` | MayaCam 1.0 camera model (Autoscoper port; verified by Lego cube reprojection) |
| `asdrr.py` | Differentiable PyTorch DRR renderer (`Bone`, `Pose` with pivot-centred increments, `View`, `masked_ncc`) |
| `filters.py` | Ports of Autoscoper filters (Sobel, contrast, Gaussian, sharpen) and `.vie` parser |
| `costs.py` | Per-frame cost (`frame_cost`), FOV mask switch, multi-bone rendering |
| `variants.py` | Cost variants: raw/current/edges/LCN, signed gradient correlation, local NCC, log transform |
| `optim.py` | Autoscoper PSO + Nelder–Mead replica and gradient refinement |
| `common.py` | Data loading (cameras, volumes, .tra row-major 4×4, radiographs, optional downsampling `DS`) |
| `fov.py` | FOV mask (per-pixel max over trial, threshold 0.2, eroded 15 px) |

## Experiments (code/) and outputs (results/, logs/)
| Script | Experiment |
|---|---|
| `holdout.py`, `holdout2.py`, `holdout_joint.py` | Per-frame hold-out: start from interpolated pose, optimise, compare to manual keyframe |
| `segtrack.py` | Keyframe-anchored gap fitting, DRR term only |
| `segtrack2.py` | **Main method**: anchored gap fitting, DRR (gradient corr.) + model-based frame-to-frame term, sine-basis corrections |
| `fwdtrack.py` | Forward-only tracking (no end anchor), sliding window + constant-velocity prediction |
| `collide.py`, `collide2.py` | Signed-distance penetration check between manual keyframe poses |
| `pcfig.py` | Point-cloud overlays on radiographs |
| `gap30.sh`, `*queue*.sh` | Run queues used for the experiments |

Result JSON names: `f2f_drr<w_drr>_f2f<w_f2f>_K<K>[_ds512]_<bones>_<a>_<b>.json`, `fwd_<bones>_<a>_<b>.json`,
`seg_<variant>_...`, `holdout*_...`.
