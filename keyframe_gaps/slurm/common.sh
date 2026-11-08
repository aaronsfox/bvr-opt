# Sourced by the sbatch scripts: environment and paths (same conda env as the methods comparison).
module load Miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bvr_opt_gpu
# Make sure the env's python is the one used (conda activate in batch scripts can leave the base first)
export PATH="$(conda env list | awk '$1 == "bvr_opt_gpu" {print $NF}')/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export BVR_RESULTS=${BVR_RESULTS:-$SLURM_SUBMIT_DIR/results/keyframe_gaps}
export PYTHONUNBUFFERED=1
mkdir -p "$BVR_RESULTS/logs"
