# Sourced by the sbatch scripts: environment and paths.
module load Miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bvr_opt_gpu
cd "$SLURM_SUBMIT_DIR"
export BVR_RESULTS=${BVR_RESULTS:-$SLURM_SUBMIT_DIR/results/methods_comparison}
export PYTHONUNBUFFERED=1
mkdir -p "$BVR_RESULTS/logs"
