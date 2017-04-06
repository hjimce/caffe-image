#!/usr/bin/env sh

/home/hjimce/tools/caffe-quantization/build/tools/caffe test \
	--model=quantized.prototxt \
	--weights=squeezenet_finetuned.caffemodel \
	--gpu=0 --iterations=2000
