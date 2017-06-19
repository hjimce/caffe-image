#!/usr/bin/env sh
TOOLS=../../caffe/build/tools
$TOOLS/caffe train -solver gender_solver.prototxt -gpu 3 2>&1 | tee log.txt 
