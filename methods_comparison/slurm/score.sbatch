#!/bin/bash
#SBATCH --job-name=mc-score
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=mc-score-%j.out
# Assemble finished units into tracks and score every arm (CPU). Submit from the repo root:
#   sbatch methods_comparison/slurm/score.sbatch
set -eo pipefail   # no -u: conda activate scripts use unset variables
source methods_comparison/slurm/common.sh
export BVR_DEVICE=cpu THREADS=$SLURM_CPUS_PER_TASK
python methods_comparison/run.py assemble
python methods_comparison/score.py
