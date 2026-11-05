#!/bin/bash
cd "$(dirname "$0")"; export THREADS=1
for a in P1 P2_drr PF; do python run_methods.py $a >> logs/$a.log 2>&1; done
