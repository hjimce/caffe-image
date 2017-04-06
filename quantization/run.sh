#!/usr/bin/env sh
TOOLS=/home/hjimce/tools/caffe-quantization/build/tools
$TOOLS/caffe train -solver solver.prototxt  -weights train.caffemodel -gpu 0  2>&1 | tee log.txt



