cd /root/work
until grep -q QUEUEDONE seg.log; do sleep 10; done
python3 holdout2.py gradcorr,log+gradcorr,log+localncc,log+lcn >> holdout.log 2>&1
echo H2DONE >> holdout.log
