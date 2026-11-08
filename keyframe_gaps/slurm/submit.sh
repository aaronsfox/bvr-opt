#!/bin/bash
# Submit one arm at one keyframe spacing as a GPU job array.
#   bash keyframe_gaps/slurm/submit.sh ARM SPACING [NJOBS] [WORKERS] [TIME] [PARTITION]
# e.g. bash keyframe_gaps/slurm/submit.sh new_opt 20
# Finished units are skipped, so a job that runs out of time can be resubmitted with the same command.
set -euo pipefail
ARM=$1; SPACING=$2; NJOBS=${3:-1}; WORKERS=${4:-4}; TIME=${5:-06:00:00}; PART=${6:-dgx}
cd "$(dirname "$0")/../.."
mkdir -p results/keyframe_gaps/logs
if [ "$PART" = dgx ]; then
  RES=(--partition=dgx --gres=gpu:1 --cpus-per-task=16 --mem=64G); DEV=cuda
else
  RES=(--partition="$PART" --cpus-per-task="$WORKERS" --mem=$(( 3 * WORKERS + 8 ))G); DEV=cpu
fi
sbatch "${RES[@]}" --array=0-$(( NJOBS - 1 )) --time="$TIME" --job-name="kg-$ARM-s$SPACING" \
  --output="results/keyframe_gaps/logs/slurm-$ARM-s$SPACING-%A_%a.out" \
  --export=ALL,ARM="$ARM",SPACING="$SPACING",WORKERS="$WORKERS",BVR_DEVICE="$DEV" keyframe_gaps/slurm/array.sbatch
