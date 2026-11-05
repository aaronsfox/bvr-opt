cd /root/work
until grep -q ALLDONE holdout.log; do sleep 10; done
python3 holdout_joint.py edges,lcn,current >> holdout.log 2>&1
echo JOINTDONE >> holdout.log
