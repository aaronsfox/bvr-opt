#!/bin/bash
# Track one trial at one or more keyframe spacings on the GPU: keyframe correction, then the gaps, then the
# optimised and spline .tra files are written (each job starts when the one before has succeeded).
#   bash general_tracking/slurm/submit.sh CFG TRA SPACING [WORKERS] [TIME]
# e.g. bash general_tracking/slurm/submit.sh datasets/UNC/Sub05_Walk1550.cfg \
#        datasets/UNC/Test_Tracking 10
#      SPACING: N, full or all; several as N:N:... (e.g. 5:10:20:full). CFG and TRA must not contain commas.
# Optional, as environment variables: BONES=Femur:Tibia  KC_TOL=1.5:2 (0:0 = no keyframe correction)  VIE=file.vie
# Finished work is skipped, so resubmitting the same command resumes.
set -euo pipefail
CFG=$1; TRA=$2; WORKERS=${4:-4}; TIME=${5:-12:00:00}
cd "$(dirname "$0")/../.."
LOGS=results/general_tracking/$(basename "$CFG" .cfg)/logs; mkdir -p "$LOGS"
GPU=(--partition=dgx --gres=gpu:1 --cpus-per-task=16 --mem=64G)
EXP="ALL,CFG=$CFG,TRA=$TRA,SPACING=${3},BONES=${BONES:-},KC_TOL=${KC_TOL:-},VIE=${VIE:-}"
SP=${3//:/-}   # spacings as they appear in job and log names: no colons (Windows cannot open them)
TAG=$(basename "$CFG" .cfg)-s$SP
DEP=()
if [ "${KC_TOL:-}" != "0:0" ]; then
  JID=$(sbatch --parsable "${GPU[@]}" --array=0-0 --time="$TIME" --job-name="gt-$TAG-keys" \
    --output="$LOGS/slurm-s$SP-keys-%A_%a.out" \
    --export="$EXP,STAGE=keys,WORKERS=1,BVR_DEVICE=cuda" general_tracking/slurm/array.sbatch)
  echo "Submitted keyframe correction: $JID"; DEP=(--dependency=afterok:$JID)
fi
JID=$(sbatch --parsable "${GPU[@]}" "${DEP[@]}" --array=0-0 --time="$TIME" --job-name="gt-$TAG-gaps" \
  --output="$LOGS/slurm-s$SP-gaps-%A_%a.out" \
  --export="$EXP,STAGE=gaps,WORKERS=$WORKERS,BVR_DEVICE=cuda" general_tracking/slurm/array.sbatch)
echo "Submitted gaps: $JID"
JID=$(sbatch --parsable --partition=normal --cpus-per-task=2 --mem=16G --time=01:00:00 --dependency=afterok:$JID \
  --job-name="gt-$TAG-assemble" --output="$LOGS/slurm-s$SP-assemble-%j.out" \
  --export="$EXP" general_tracking/slurm/assemble.sbatch)
echo "Submitted assemble: $JID"
