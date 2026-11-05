#!/bin/bash
#SBATCH --job-name=a1
# Runs one arm's units, spread over the array tasks and WORKERS processes per job (sharing the job's GPU,
# or one CPU thread each on a CPU partition). Partition, GPU and CPUs are set by submit.sh.
set -eo pipefail   # no -u: conda activate scripts use unset variables
source analysis1/slurm/common.sh
WORKERS=${WORKERS:-4}
NT=$(( SLURM_ARRAY_TASK_COUNT * WORKERS ))
export THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS > 0 ? SLURM_CPUS_PER_TASK / WORKERS : 1 ))
for j in $(seq 0 $(( WORKERS - 1 ))); do
  T=$(( SLURM_ARRAY_TASK_ID * WORKERS + j ))
  python analysis1/run.py run --arm "$ARM" --task $T --ntasks $NT \
    > "$BVR_RESULTS/logs/${ARM//,/+}_${SLURM_ARRAY_JOB_ID}_${T}.log" 2>&1 &
done
wait
