#!/bin/bash
#SBATCH --job-name=mc-setup
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=mc-setup-%j.out
# Batch version of setup_env.sh (creates the conda env `bvr_opt_gpu`). Submit from the repo root:
#   sbatch methods_comparison/slurm/setup_env.sbatch
# Needs internet access from the compute node; if it fails to download, run
# `bash methods_comparison/slurm/setup_env.sh` on the login node instead.
cd "$SLURM_SUBMIT_DIR"
bash methods_comparison/slurm/setup_env.sh
