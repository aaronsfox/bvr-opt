#!/bin/bash
# Submit one arm at one keyframe spacing as a GPU job array.
#   bash keyframe_gaps/slurm/submit.sh ARM SPACING [NJOBS] [WORKERS] [TIME] [PARTITION]
# e.g. bash keyframe_gaps/slurm/submit.sh new_opt 20
# Finished units are skipped, so a job that runs out of time can be resubmitted with the same command.
# Arms with keyframe correction (e.g. new_opt_stepwise_kc) are submitted as two jobs: the keyframe-correction
# stage (one GPU, one worker per trial), then the gaps, which start only when that stage has succeeded.
set -euo pipefail
ARM=$1; SPACING=$2; NJOBS=${3:-1}; WORKERS=${4:-4}; TIME=${5:-06:00:00}; PART=${6:-dgx}
cd "$(dirname "$0")/../.."
mkdir -p results/keyframe_gaps/logs
if [ "$PART" = dgx ]; then
  RES=(--partition=dgx --gres=gpu:1 --cpus-per-task=16 --mem=64G); DEV=cuda
else
  RES=(--partition="$PART" --cpus-per-task="$WORKERS" --mem=$(( 3 * WORKERS + 8 ))G); DEV=cpu
fi
DEP=()
if [ "$(BVR_DEVICE=cpu python keyframe_gaps/run.py has-kc --arm "$ARM")" = yes ]; then
  JID=$(sbatch --parsable "${RES[@]}" --array=0-0 --time="$TIME" --job-name="kg-$ARM-s$SPACING-keys" \
    --output="results/keyframe_gaps/logs/slurm-$ARM-s$SPACING-keys-%A_%a.out" \
    --export=ALL,ARM="$ARM",SPACING="$SPACING",STAGE=keys,WORKERS=4,BVR_DEVICE="$DEV" keyframe_gaps/slurm/array.sbatch)
  echo "Submitted keyframe-correction job $JID"
  DEP=(--dependency=afterok:$JID)
fi
sbatch "${RES[@]}" "${DEP[@]}" --array=0-$(( NJOBS - 1 )) --time="$TIME" --job-name="kg-$ARM-s$SPACING" \
  --output="results/keyframe_gaps/logs/slurm-$ARM-s$SPACING-%A_%a.out" \
  --export=ALL,ARM="$ARM",SPACING="$SPACING",STAGE=gaps,WORKERS="$WORKERS",BVR_DEVICE="$DEV" keyframe_gaps/slurm/array.sbatch
