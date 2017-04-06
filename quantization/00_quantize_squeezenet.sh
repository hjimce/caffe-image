#!/usr/bin/env sh

/home/hjimce/tools/caffe-quantization/build/tools/ristretto quantize \
	--model=train_val.prototxt \
	--weights=train.caffemodel \
	--model_quantized=quantized.prototxt \
	--trimming_mode=dynamic_fixed_point --gpu=0 --iterations=3000 \
	--error_margin=3
