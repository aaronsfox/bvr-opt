cd /root/work
until grep -q JOINTDONE holdout.log; do sleep 10; done
python3 segtrack.py edges 166 176 1stMet,2ndMet 171
python3 segtrack.py edges 166 176 2ndMet 171
python3 segtrack.py lcn 146 166 1stPhal 156
python3 segtrack.py lcn 156 176 1stPhal 166
python3 segtrack.py edges 146 166 1stMet,2ndMet 151,156,161
python3 segtrack.py lcn 156 166 1stMet 161
echo QUEUEDONE
