cd /root/work
G1="166 176 1stMet,2ndMet 171"; G2="146 166 1stMet,2ndMet 151,156,161"; G3="146 166 1stPhal 156"; G4="156 176 1stPhal 166"
for g in "$G1" "$G2" "$G3" "$G4"; do python3 segtrack2.py 0 1 $g; done
for g in "$G1" "$G3" "$G4" "$G2"; do python3 segtrack2.py 1 2 $g; done
for g in "$G1" "$G3" "$G4" "$G2"; do python3 segtrack2.py 1 0 $g; done
echo F2FQUEUEDONE
