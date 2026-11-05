#!/bin/bash
# Create the conda environment `bvr_opt_gpu` used by the SLURM scripts (run once; it needs internet access).
# Following environment.yml, run it on a compute node:
#   srun -p normal -c 4 --mem=16G -t 01:00:00 bash analysis1/slurm/setup_env.sh
# (if compute nodes have no internet, run `bash analysis1/slurm/setup_env.sh` on the login node instead).
# The CUDA build of PyTorch also runs on CPU-only nodes, so the same environment serves both partitions.
# PyTorch comes from the CUDA 12.6 wheel index, which runs on NVIDIA drivers >= 525 (CUDA 12.x minor-version
# compatibility). If the smoke test reports an old driver, change CU below (cu118 for drivers >= 450).
set -eo pipefail   # no -u: conda activate scripts use unset variables
CU=${CU:-cu126}
cd "$(dirname "$0")/../.."
module load Miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n bvr_opt_gpu -c conda-forge --override-channels python=3.12 pip
conda activate bvr_opt_gpu
pip install torch --index-url https://download.pytorch.org/whl/$CU
pip install -r requirements.txt          # torch already satisfied, so the CUDA build is kept
python -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"
