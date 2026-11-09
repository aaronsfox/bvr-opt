#!/bin/bash
#SBATCH --job-name=rg
# One simulated rater (array index = rater id) at one level and one spacing, for one stage (keys or gaps).
# WORKERS processes share the job's GPU and split the stage's units. Use submit.sh.
set -eo pipefail   # no -u: conda activate scripts use unset variables
source rater_gaps/slurm/common.sh
R=$(printf "r%02d" "$SLURM_ARRAY_TASK_ID")
export BVR_RESULTS="$SLURM_SUBMIT_DIR/results/rater_gaps/L${LEVEL}/${R}"
mkdir -p "$BVR_RESULTS"
WORKERS=${WORKERS:-4}
export THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS > 0 ? SLURM_CPUS_PER_TASK / WORKERS : 1 ))
PIDS=()
for j in $(seq 0 $(( WORKERS - 1 ))); do
  python keyframe_gaps/run.py run --arm "$ARM" --spacing "$SPACING" --stage "$STAGE" --task $j --ntasks $WORKERS \
    --keyframes "rater:L${LEVEL}/${R}" --kc-tol "$TOL" \
    > "results/rater_gaps/logs/L${LEVEL}_${R}_s${SPACING}_${STAGE}_${SLURM_JOB_ID}_${j}.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
if [ $FAIL -ne 0 ]; then echo "a worker failed: see results/rater_gaps/logs/L${LEVEL}_${R}_s${SPACING}_${STAGE}_${SLURM_JOB_ID}_*.log"; exit 1; fi
