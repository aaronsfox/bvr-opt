#!/bin/bash
cd "$(dirname "$0")"; export THREADS=1
for a in S2 P2 P3 P2_f2f P3tol; do python run_methods.py $a >> logs/$a.log 2>&1; done
