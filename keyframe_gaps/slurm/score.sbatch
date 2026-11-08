#!/bin/bash
#SBATCH --job-name=kg-score
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=kg-score-%j.out
# Interpolation reference, assembly and scoring (CPU). Submit from the repo root:
#   sbatch keyframe_gaps/slurm/score.sbatch
set -eo pipefail
source keyframe_gaps/slurm/common.sh
export BVR_DEVICE=cpu THREADS=$SLURM_CPUS_PER_TASK
python keyframe_gaps/run.py run --arm keyframe_interp
python keyframe_gaps/run.py assemble
python keyframe_gaps/score.py
