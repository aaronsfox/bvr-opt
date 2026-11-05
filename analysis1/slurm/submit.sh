#!/bin/bash
# Submit one arm (or a comma-separated list of arms) as a job array.
#   bash analysis1/slurm/submit.sh ARM NJOBS [WORKERS] [TIME] [PARTITION]
# GPU (default):  bash analysis1/slurm/submit.sh B 24 4 48:00:00          -> dgx, 1 GPU and 16 CPUs per job
# CPU:            bash analysis1/slurm/submit.sh B 4 48 48:00:00 normal   -> normal, WORKERS single-thread processes per job
# Units are spread over NJOBS x WORKERS processes; finished units are skipped, so a job that runs out of
# time can simply be resubmitted. List the units with: python analysis1/run.py list --arm ARM
set -euo pipefail
ARM=$1; NJOBS=$2; WORKERS=${3:-4}; TIME=${4:-24:00:00}; PART=${5:-dgx}
cd "$(dirname "$0")/../.."
mkdir -p results/analysis1/logs
if [ "$PART" = dgx ]; then
  RES=(--partition=dgx --gres=gpu:1 --cpus-per-task=16 --mem=64G); DEV=cuda
else
  RES=(--partition="$PART" --cpus-per-task="$WORKERS" --mem=$(( 3 * WORKERS + 8 ))G); DEV=cpu
fi
sbatch "${RES[@]}" --array=0-$(( NJOBS - 1 )) --time="$TIME" --job-name="a1-${ARM//,/+}" \
  --output="results/analysis1/logs/slurm-${ARM//,/+}-%A_%a.out" \
  --export=ALL,ARM="$ARM",WORKERS="$WORKERS",BVR_DEVICE="$DEV" analysis1/slurm/array.sbatch
