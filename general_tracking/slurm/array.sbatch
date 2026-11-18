#!/bin/bash
#SBATCH --job-name=gt
# One stage (keys or gaps) for one trial and spacing; the gaps are shared between WORKERS processes on the
# job's GPU. Partition, GPU and CPUs are set by submit.sh.
set -eo pipefail   # no -u: conda activate scripts use unset variables
source general_tracking/slurm/common.sh
WORKERS=${WORKERS:-4}
NT=$(( SLURM_ARRAY_TASK_COUNT * WORKERS ))
export THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS > 0 ? SLURM_CPUS_PER_TASK / WORKERS : 1 ))
PIDS=()
for j in $(seq 0 $(( WORKERS - 1 ))); do
  T=$(( SLURM_ARRAY_TASK_ID * WORKERS + j ))
  python general_tracking/run.py run "${ARGS[@]}" --stage "$STAGE" --task $T --ntasks $NT \
    > "$BVR_RESULTS/logs/s${SPACING}_${STAGE}_${SLURM_ARRAY_JOB_ID}_${T}.log" 2>&1 &
  PIDS+=($!)
done
# Fail the job (state FAILED in sacct) if any worker exits with an error, so crashes are not hidden
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
if [ $FAIL -ne 0 ]; then echo "a worker failed: see $BVR_RESULTS/logs/s${SPACING}_${STAGE}_${SLURM_ARRAY_JOB_ID}_*.log"; exit 1; fi
