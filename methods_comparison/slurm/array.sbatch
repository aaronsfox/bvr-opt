#!/bin/bash
#SBATCH --job-name=a1
# Runs one arm's units, spread over the array tasks and WORKERS processes per job (sharing the job's GPU,
# or one CPU thread each on a CPU partition). Partition, GPU and CPUs are set by submit.sh.
set -eo pipefail   # no -u: conda activate scripts use unset variables
source methods_comparison/slurm/common.sh
WORKERS=${WORKERS:-4}
NT=$(( SLURM_ARRAY_TASK_COUNT * WORKERS ))
export THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS > 0 ? SLURM_CPUS_PER_TASK / WORKERS : 1 ))
PIDS=()
for j in $(seq 0 $(( WORKERS - 1 ))); do
  T=$(( SLURM_ARRAY_TASK_ID * WORKERS + j ))
  python methods_comparison/run.py run --arm "$ARM" --task $T --ntasks $NT \
    > "$BVR_RESULTS/logs/${ARM//,/+}_${SLURM_ARRAY_JOB_ID}_${T}.log" 2>&1 &
  PIDS+=($!)
done
# Fail the job (state FAILED in sacct) if any worker exits with an error, so crashes are not hidden
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
if [ $FAIL -ne 0 ]; then echo "a worker failed: see $BVR_RESULTS/logs/${ARM//,/+}_${SLURM_ARRAY_JOB_ID}_*.log"; exit 1; fi
