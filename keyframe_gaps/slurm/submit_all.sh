#!/bin/bash
# Submit one job array per keyframe spacing for an arm (default new_opt): six small, independent jobs.
#   bash keyframe_gaps/slurm/submit_all.sh [ARM]
# Jobs per spacing are sized to the number of units (5 -> 36 gaps, full -> 4 gaps); 4 workers share each GPU.
set -euo pipefail
ARM=${1:-new_opt}
cd "$(dirname "$0")/../.."
bash keyframe_gaps/slurm/submit.sh "$ARM" 5    2 4 06:00:00
bash keyframe_gaps/slurm/submit.sh "$ARM" 10   1 4 06:00:00
bash keyframe_gaps/slurm/submit.sh "$ARM" 20   1 4 08:00:00
bash keyframe_gaps/slurm/submit.sh "$ARM" 30   1 4 08:00:00
bash keyframe_gaps/slurm/submit.sh "$ARM" 40   1 4 08:00:00
bash keyframe_gaps/slurm/submit.sh "$ARM" full 1 4 08:00:00
