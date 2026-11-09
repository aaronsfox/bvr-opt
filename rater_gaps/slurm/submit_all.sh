#!/bin/bash
# Submit every level x spacing (3 x 6 = 18 pairs of small arrays, 10 raters each).
#   bash rater_gaps/slurm/submit_all.sh [MAX_CONCURRENT_PER_ARRAY]
# Every array task is one rater on one GPU for one stage (minutes to about an hour), so tasks slot into gaps
# on dgx. MAX_CONCURRENT_PER_ARRAY (e.g. 2) limits how many of each array's tasks run at once.
set -euo pipefail
MAXC=${1:-}
cd "$(dirname "$0")/../.."
for L in 1 2 4; do
  for S in 5 10 20 30 40 full; do
    bash rater_gaps/slurm/submit.sh "$L" "$S" 0-9 04:00:00 $MAXC
  done
done
