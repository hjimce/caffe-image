#!/usr/bin/env sh
TOOLS=/home/research/tools/caffe/build/tools
$TOOLS/caffe train -solver gender_solver.prototxt -gpu 1 2>&1 | tee log.txt 
