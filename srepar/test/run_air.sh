#!/bin/bash

PYPCKG="srepar.examples.air.run"
ARGS="-n 50000 -ef 10 -blr 0.1 --z-pres-prior 0.01" # same as `--progress-every 10`.
I=1

for ((i=0;i<$I;i++)); do
    LS=_$(date "+%Y%m%d-%H%M%S")
    (cd ..; python3 -m $PYPCKG -r ours  -ls $LS $ARGS) #&
    (cd ..; python3 -m $PYPCKG -r repar -ls $LS $ARGS) #&
    (cd ..; python3 -m $PYPCKG -r score -ls $LS $ARGS) #&
    wait
done
