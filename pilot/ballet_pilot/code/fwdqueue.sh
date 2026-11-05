cd /root/work
until grep -q DSQUEUEDONE ds.log; do sleep 10; done
python3 fwdtrack.py 146 180 1stMet,2ndMet 1 2 0.02 3
python3 fwdtrack.py 146 176 1stPhal 1 2 0.02 3
echo FWDQUEUEDONE
