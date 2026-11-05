cd /root/work
for v in edges current lcn raw; do python3 holdout.py grad:$v; done
python3 holdout.py pso:current 1stMet:151,1stMet:156,1stMet:161,1stMet:166,1stMet:171
echo ALLDONE
