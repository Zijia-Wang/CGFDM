#!/usr/bin/bash

#set -x
set -e

#python config_params.py

PX=`python get_params.py PX`
PY=`python get_params.py PY`
PZ=`python get_params.py PZ`
OUT=`python get_params.py OUT`


#NP=`echo "$PX*$PY*$PZ"|bc -q`
NP=`echo $PX $PY $PZ|awk '{print $1*$2*$3}'`
echo NP=$NP

echo "teslagpus slots=8" > nodelists

EXE="../../bin/a.out"

#rm -rf $OUT && mkdir -p $OUT
mkdir -p $OUT
#cp params.json $OUT

RUN=/public/software/openmpi-4.1.1-cuda.10/bin/mpirun

if [ $NP -eq 1 ];then
echo "serial"
${EXE}
else
$RUN -np $NP -x LD_LIBRARY_PATH --machinefile nodelists ${EXE} 2>&1|tee log
fi
