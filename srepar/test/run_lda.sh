#!/bin/bash

PYPCKG="srepar.examples.lda.run"
ARGS="-n 1000 -ef 10 -lr 1e-2"
I=1

for ((i=0;i<$I;i++)); do
    LS=_$(date "+%Y%m%d-%H%M%S")
    (cd ..; python3 -m $PYPCKG -r ours  -ls $LS $ARGS) #&
    (cd ..; python3 -m $PYPCKG -r repar -ls $LS $ARGS) #&
    (cd ..; python3 -m $PYPCKG -r score -ls $LS $ARGS) #&
    wait
done
