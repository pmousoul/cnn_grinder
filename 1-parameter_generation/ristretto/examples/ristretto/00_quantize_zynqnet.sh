#!/usr/bin/env sh

./build/tools/ristretto quantize \
	--model=models/ZynqNet/train_val.prototxt \
	--weights=models/ZynqNet/zynqnet.caffemodel \
	--model_quantized=models/ZynqNet/RistrettoDemo/quantized.prototxt \
	--trimming_mode=dynamic_fixed_point --gpu=0 --iterations=2000 \
	--error_margin=3
