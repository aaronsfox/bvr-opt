#!/bin/bash
#SBATCH --job-name=a1-smoke
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=a1-smoke-%j.out
# Checks the GPU environment and times each arm. Submit from the repo root:
#   sbatch analysis1/slurm/smoke.sbatch
# Nothing is written to the results folder; read the timings at the end of a1-smoke-<jobid>.out.
set -eo pipefail   # no -u: conda activate scripts use unset variables
source analysis1/slurm/common.sh
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '')"
python analysis1/bench.py
export THREADS=$SLURM_CPUS_PER_TASK
python analysis1/run.py run --arm B --quick --max-units 1
python analysis1/run.py run --arm C --quick --max-units 1
python analysis1/run.py run --arm C_perframe --quick --max-units 1
