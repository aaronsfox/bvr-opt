#!/bin/bash
#SBATCH --job-name=rg-kc-step-test
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=rg-kc-step-test-%j.out
# Keyframe-correction step test on one GPU: original vs fixed-size steps, for level-4 raters r00 and r01 and
# level-1 rater r00, all keyframes at spacing `full` (2 per trial, 8 per rater). Submit from the repo root:
#   sbatch rater_gaps/slurm/kc_step_test.sbatch
# The comparison table is printed at the end of rg-kc-step-test-<jobid>.out and saved in
# results/rater_gaps/tests/kc_step/summary.md.
set -eo pipefail
source rater_gaps/slurm/common.sh
export BVR_DEVICE=cuda THREADS=2
PIDS=()
for cfg in "4 r00" "4 r01" "1 r00"; do
  set -- $cfg
  for st in tol mm; do
    python rater_gaps/tests/kc_step_test.py run --step $st --level $1 --rater $2 --spacing full \
      > results/rater_gaps/logs/kc_step_${st}_L$1_$2.log 2>&1 &
    PIDS+=($!)
  done
done
FAIL=0; for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
python rater_gaps/tests/kc_step_test.py summary
exit $FAIL
