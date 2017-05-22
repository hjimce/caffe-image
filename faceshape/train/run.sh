#!/usr/bin/env sh
TOOLS=/home/research/tools/caffe/build/tools
$TOOLS/caffe train -solver solver.prototxt  -gpu 0 -weights iter_iter_5000.caffemodel 2>&1 | tee log.txt



