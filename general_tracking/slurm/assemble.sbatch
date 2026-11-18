#!/bin/bash
#SBATCH --job-name=gt-assemble
# Writes the optimised and spline .tra files once the gaps are done (CPU, about a minute).
set -eo pipefail
source general_tracking/slurm/common.sh
BVR_DEVICE=cpu python general_tracking/run.py assemble "${ARGS[@]}"
