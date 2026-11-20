# Sourced by the sbatch scripts: environment (same conda env as the methods comparison) and arguments.
module load Miniconda3/4.9.2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bvr_opt_gpu
# Make sure the env's python is the one used (conda activate in batch scripts can leave the base first)
export PATH="$(conda env list | awk '$1 == "bvr_opt_gpu" {print $NF}')/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
export BVR_RESULTS=${BVR_RESULTS:-$SLURM_SUBMIT_DIR/results/general_tracking/$(basename "$CFG" .cfg)}
export PYTHONUNBUFFERED=1
mkdir -p "$BVR_RESULTS/logs"
# Arguments for run.py (optional values are colon-separated: sbatch --export splits at commas)
ARGS=(--cfg "$CFG" --tra "$TRA" --spacing "${SPACING//:/,}")
# (if-blocks, not `[ ] && ...`: a false test as the last command would make `source` fail under set -e)
if [ -n "${BONES:-}" ]; then ARGS+=(--bones "$BONES"); fi
if [ -n "${KC_TOL:-}" ]; then ARGS+=(--kc-tol "$KC_TOL"); fi
if [ -n "${VIE:-}" ]; then ARGS+=(--vie "$VIE"); fi
if [ -n "${TRA_PATTERN:-}" ]; then ARGS+=(--tra-pattern "$TRA_PATTERN"); fi
if [ -n "${PER_BONE:-}" ]; then ARGS+=(--per-bone); fi
if [ -n "${D_MIN:-}" ]; then ARGS+=("--d-min=$D_MIN"); fi
if [ -n "${MU:-}" ]; then ARGS+=("--mu=$MU"); fi
