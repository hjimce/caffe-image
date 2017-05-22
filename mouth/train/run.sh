#!/usr/bin/env sh
TOOLS=/home/research/tools/caffe/build/tools
$TOOLS/caffe train -solver gender_solver.prototxt  -gpu 0  2>&1 | tee log.txt



