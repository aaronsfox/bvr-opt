#!/bin/bash
# per-frame PSO (S3) on the gap midpoints, one job per frame; start each when a queue frees a core
cd "$(dirname "$0")"; export THREADS=1
python run_methods.py S3$1 $2 >> logs/S3$1.log 2>&1
