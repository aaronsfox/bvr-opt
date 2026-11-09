# Sourced by the sbatch scripts: environment (same conda env as the other analyses).
module load Miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bvr_opt_gpu
export PATH="$(conda env list | awk '$1 == "bvr_opt_gpu" {print $NF}')/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
mkdir -p results/rater_gaps/logs
