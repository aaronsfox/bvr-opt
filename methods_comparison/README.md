# Methods comparison with perfect keyframes

Analysis 1 in the plan doc (Study design). All four SOL001A trials, tibia, talus and calcaneus. Keyframes
are the bead-based reference poses every 10 frames, at four offsets (first keyframe at frame 0, 3, 5, 8).
Frames between the first and last keyframe are scored against the reference; keyframes are excluded.

| Arm | What it is | Units (one SLURM work item each) |
|---|---|---|
| **keyframe_interp** | Spline interpolation through the keyframes (pivot cubic spline + rotation spline) | 16 (trial × offset) |
| **pso_simplex** | Autoscoper forward pass, restarted at every keyframe: each frame starts from the previous frame's result, PSO + simplex at Autoscoper defaults, filtered NCC with the Welte `.vie` filters, full resolution (render 1760, 2048² radiographs) | 222 (trial × offset × bone × gap) |
| **new_opt** | New method: keyframe-anchored gap fit, tibia + talus + calcaneus co-fitted, signed gradient correlation + frame-to-frame + collision (PV, d_min −2 mm), hard anchors | 74 (trial × offset × gap) |
| keyframe_interp_linear | Supplementary: linear + slerp interpolation | 16 |
| pso_simplex_512 | Supplementary: pso_simplex at 512² | 222 |
| pso_simplex_single_pass | Supplementary: Akhbari 2021 literally, one forward pass from the first keyframe only (offset 0) | 12 |
| new_opt_single_bone, new_opt_drr_only, new_opt_f2f_only, new_opt_no_collision | Supplementary ablations of new_opt: bones fitted alone; DRR term only; frame-to-frame only; no collision | 74 each |
| new_opt_per_frame | Supplementary ablation: no anchoring, each frame fitted on its own | 74 |

Settings live in `config.yaml`; `python methods_comparison/run.py list` prints every unit. The package
used by the pipeline is `code/bvr_opt`.

## Running on the HPC (SLURM)

From the repo root on the login node:

```bash
# 1. once: conda env `bvr_opt_gpu` (batch job on `normal`; read mc-setup-<jobid>.out)
sbatch methods_comparison/slurm/setup_env.sbatch

# 2. smoke test on `dgx` after the env is built: checks the GPU and times the expensive arms (nothing saved)
sbatch --dependency=afterok:<setup jobid> methods_comparison/slurm/smoke.sbatch    # read mc-smoke-<jobid>.out

# 3. submit arms as job arrays: ARM NJOBS [WORKERS_PER_JOB] [TIME] [PARTITION (default dgx)]
python methods_comparison/run.py run --arm keyframe_interp,keyframe_interp_linear    # seconds, no GPU needed
bash methods_comparison/slurm/submit.sh new_opt 8 4 12:00:00
bash methods_comparison/slurm/submit.sh pso_simplex 24 4 48:00:00
bash methods_comparison/slurm/submit.sh new_opt_single_bone,new_opt_drr_only,new_opt_f2f_only,new_opt_no_collision,new_opt_per_frame 8 4 24:00:00

# 4. progress: units finished per arm
for a in keyframe_interp pso_simplex new_opt; do echo $a $(ls results/methods_comparison/$a/units 2>/dev/null | wc -l); done
squeue -u $USER

# 5. assemble and score (CPU; fine to run part-way too)
sbatch methods_comparison/slurm/score.sbatch     # -> results/methods_comparison/summary.md, scores.csv, figures/
```

Size NJOBS and TIME from the smoke-test estimate. Every unit is written when it finishes and skipped
when rerun, so a job that hits its time limit is simply resubmitted with the same command. Several worker
processes share each GPU (`WORKERS`, default 4), since a single PSO run keeps a GPU only partly busy. On a
CPU partition (`... normal`), WORKERS is the number of single-thread processes per job.

## Results

Written to `results/methods_comparison/` in the repo (or `$BVR_RESULTS`), one folder per arm:

```
results/methods_comparison/
  keyframe_interp/  pso_simplex/  new_opt/  ...   one folder per arm
    units/*.json                  one file per unit: poses, per-frame costs, timings, device
    tracks/<trial>_o<offset>.npz  after `assemble`: {bone: (n_frames, 4, 4)}
  scores.csv  summary.md  figures/                after `score.py`
  logs/                                           per-process logs (not committed)
```

Commit and push the results folder from the cluster when the runs finish.

## Outcomes

`score.py` reports, per arm, pooled over trials and offsets: median and 95th percentile of pivot error
and helical rotation per bone, talus–calcaneus relative rotation, and the share of bone-frames that miss
Welte's operator standard (pivot error > 1 mm or rotation > 2°), i.e. frames an operator would expect to
correct. Unfiltered is primary; the same 10 Hz Butterworth filter on every arm is secondary.

## Device

`BVR_DEVICE=cuda|cpu` (default: CUDA when available). Every arm also runs on the CPU, just slowly.
