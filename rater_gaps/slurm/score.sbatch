#!/bin/bash
#SBATCH --job-name=rg-score
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=rg-score-%j.out
# Interpolation through each rater's keyframes, assembly and scoring (CPU). Submit from the repo root:
#   sbatch rater_gaps/slurm/score.sbatch
set -eo pipefail
source rater_gaps/slurm/common.sh
export BVR_DEVICE=cpu THREADS=$SLURM_CPUS_PER_TASK
python rater_gaps/score.py --interp --assemble
