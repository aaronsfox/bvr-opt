#!/bin/bash
#SBATCH --job-name=a1-score
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=a1-score-%j.out
# Assemble finished units into tracks and score every arm (CPU). Submit from the repo root:
#   sbatch analysis1/slurm/score.sbatch
set -eo pipefail   # no -u: conda activate scripts use unset variables
source analysis1/slurm/common.sh
export BVR_DEVICE=cpu THREADS=$SLURM_CPUS_PER_TASK
python analysis1/run.py assemble
python analysis1/score.py
