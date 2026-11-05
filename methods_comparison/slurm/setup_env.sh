#!/bin/bash
# Create the conda environment `bvr_opt_gpu` used by the SLURM scripts (run once; it needs internet access).
# Following environment.yml, run it on a compute node:
#   sbatch methods_comparison/slurm/setup_env.sbatch       (batch job; or `bash` this script on a node)
# (if compute nodes have no internet, run `bash methods_comparison/slurm/setup_env.sh` on the login node instead).
# The CUDA build of PyTorch also runs on CPU-only nodes, so the same environment serves both partitions.
# PyTorch comes from the CUDA 12.6 wheel index, which runs on NVIDIA drivers >= 525 (CUDA 12.x minor-version
# compatibility). If the smoke test reports an old driver, change CU below (cu118 for drivers >= 450).
set -eo pipefail   # no -u: conda activate scripts use unset variables
CU=${CU:-cu126}
cd "$(dirname "$0")/../.."
module load Miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
# Create the env unless it already exists (a rerun after a failed install reuses it).
if ! conda env list | awk '{print $1}' | grep -qx bvr_opt_gpu; then
  conda create -y -n bvr_opt_gpu -c conda-forge --override-channels python=3.12 pip
fi
# Call the env's own python directly: `conda activate` inside a batch script can leave the module's
# base Python (3.8) first on PATH, which has no matching torch wheels.
ENVDIR=$(conda env list | awk '$1 == "bvr_opt_gpu" {print $NF}')
PY="$ENVDIR/bin/python"
"$PY" --version
"$PY" -m pip install torch --index-url https://download.pytorch.org/whl/$CU
"$PY" -m pip install -r requirements.txt          # torch already satisfied, so the CUDA build is kept
"$PY" -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)"
