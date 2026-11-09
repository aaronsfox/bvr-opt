#!/bin/bash
# Submit one rater level at one keyframe spacing: two small job arrays, one task per rater (1 GPU each).
#   bash rater_gaps/slurm/submit.sh LEVEL SPACING [RATERS] [TIME] [MAX_CONCURRENT]
# e.g. bash rater_gaps/slurm/submit.sh 2 20            # raters 0-9, level 2, spacing 20
#      bash rater_gaps/slurm/submit.sh 1 10 0-4 04:00:00 3
# Stage 1 (keys): keyframe correction of that rater's keyframes. Stage 2 (gaps): the stepwise fit; each
# rater's gaps task starts as soon as the same rater's keys task has succeeded (aftercorr).
# Finished units are skipped, so resubmitting the same command only redoes what is missing.
set -euo pipefail
LEVEL=$1; SPACING=$2; RATERS=${3:-0-9}; TIME=${4:-04:00:00}; MAXC=${5:-}
cd "$(dirname "$0")/../.."
mkdir -p results/rater_gaps/logs
ARM=$(python -c "import yaml; print(yaml.safe_load(open('rater_gaps/config.yaml'))['arm'])")
# mm:deg with a colon: sbatch --export splits its value on commas
TOL=$(python -c "import yaml; t=yaml.safe_load(open('rater_gaps/config.yaml'))['kc_tol'][$LEVEL]; print(f'{t[0]}:{t[1]}')")
ARR="$RATERS${MAXC:+%$MAXC}"
RES=(--partition=dgx --gres=gpu:1 --cpus-per-task=16 --mem=64G --time="$TIME")
EXP="ALL,ARM=$ARM,LEVEL=$LEVEL,SPACING=$SPACING,TOL=$TOL,WORKERS=4,BVR_DEVICE=cuda"
K=$(sbatch --parsable "${RES[@]}" --array="$ARR" --job-name="rg-L$LEVEL-s$SPACING-keys" \
  --output="results/rater_gaps/logs/slurm-L$LEVEL-s$SPACING-keys-%A_%a.out" --export="$EXP,STAGE=keys" rater_gaps/slurm/array.sbatch)
G=$(sbatch --parsable "${RES[@]}" --array="$ARR" --dependency=aftercorr:$K --job-name="rg-L$LEVEL-s$SPACING-gaps" \
  --output="results/rater_gaps/logs/slurm-L$LEVEL-s$SPACING-gaps-%A_%a.out" --export="$EXP,STAGE=gaps" rater_gaps/slurm/array.sbatch)
echo "level $LEVEL spacing $SPACING raters $RATERS: keys $K, gaps $G"
