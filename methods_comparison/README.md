# Methods comparison with perfect keyframes

All four SOL001A trials, tibia, talus and calcaneus. Keyframes are the bead-based reference poses every
10 frames, at four offsets (first keyframe at frame 0, 3, 5, 8). Frames between the first and last
keyframe are scored against the reference; keyframes are excluded. Design: plan doc, Study design (Analysis 1).

| Arm | What it is | Units (one SLURM work item each) |
|---|---|---|
| **A** | Spline interpolation through the keyframes (pivot cubic spline + rotation spline) | 16 (trial × offset) |
| **B** | Autoscoper forward pass, restarted at every keyframe: each frame starts from the previous frame's result, PSO + simplex at Autoscoper defaults, filtered NCC with the Welte `.vie` filters, full resolution (render 1760, 2048² radiographs) | 222 (trial × offset × bone × gap) |
| **C** | New method: keyframe-anchored gap fit, tibia + talus + calcaneus co-fitted, signed gradient correlation + frame-to-frame + collision (PV, d_min −2 mm), hard anchors | 74 (trial × offset × gap) |
| A_linear | Supplementary: linear + slerp interpolation | 16 |
| B_512 | Supplementary: arm B at 512² | 222 |
| B_akhbari | Supplementary: Akhbari 2021 literally, one forward pass from the first keyframe only (offset 0) | 12 |
| C_alone, C_drr, C_f2f, C_nocontact | Supplementary ablations of C: bones alone; DRR term only; frame-to-frame only; no collision | 74 each |
| C_perframe | Supplementary ablation: no anchoring, each frame fitted on its own | 74 |

Settings live in `config.yaml`; `python methods_comparison/run.py list` prints every unit.

## Running on the HPC (SLURM, `dgx` partition)

From the repo root on the login node:

```bash
# 1. once: GPU conda environment (needs internet, so run it on the login node)
bash methods_comparison/slurm/setup_env.sh

# 2. smoke test: checks the GPU, times arms B and C (nothing is saved). Read the end of mc-smoke-<jobid>.out
sbatch methods_comparison/slurm/smoke.sbatch

# 3. submit the arms as job arrays: ARM NJOBS [WORKERS_PER_GPU] [TIME]
bash methods_comparison/slurm/submit.sh C 8 4 12:00:00
bash methods_comparison/slurm/submit.sh B 24 4 48:00:00
bash methods_comparison/slurm/submit.sh C_alone,C_drr,C_f2f,C_nocontact,C_perframe 8 4 24:00:00

# 4. progress: units finished per arm
for a in A B C; do echo $a $(ls results/methods_comparison/$a/units 2>/dev/null | wc -l); done
squeue -u $USER

# 5. assemble and score (CPU, after the arrays finish; also fine to run part-way)
sbatch methods_comparison/slurm/score.sbatch      # -> results/methods_comparison/summary.md, scores.csv, figures/
```

Arm A needs no GPU (`python methods_comparison/run.py run --arm A,A_linear`, a few seconds). Size NJOBS and
TIME from the smoke-test estimate. Every unit is written when it finishes and skipped when rerun, so a job
that hits its time limit is simply resubmitted with the same command. Several worker processes share
each GPU (`WORKERS`, default 4), since a single PSO run keeps a GPU only partly busy.

Results go to `results/methods_comparison/` in the repo (or `$BVR_RESULTS`), one folder per arm:

```
results/methods_comparison/
  A/  B/  C/  ...            one folder per arm
    units/*.json             one file per unit: poses, per-frame costs, timings, device
    tracks/<trial>_o<offset>.npz   after `assemble`: {bone: (n_frames, 4, 4)}
  scores.csv  summary.md  figures/   after `score.py`
  logs/                      per-process logs (not committed)
```

Commit and push the results folder from the cluster when the runs finish (logs are git-ignored).

## Outcomes

`score.py` reports, per arm, pooled over trials and offsets: median and 95th percentile of pivot error
and helical rotation per bone, talus–calcaneus relative rotation, and the share of bone-frames that miss
Welte's operator standard (pivot error > 1 mm or rotation > 2°), i.e. frames an operator would expect to
correct. Unfiltered is primary; the same 10 Hz Butterworth filter on every arm is secondary.

## Device

`BVR_DEVICE=cuda|cpu` (default: CUDA when available). Every arm also runs on the CPU, just slowly.
