#!/usr/bin/env sh

./build/tools/ristretto quantize \
	--model=models/SqueezeNet_v1.1/train_val.prototxt \
	--weights=models/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel \
	--model_quantized=models/SqueezeNet_v1.1/RistrettoDemo/quantized.prototxt \
	--trimming_mode=dynamic_fixed_point --gpu=0 --iterations=2000 \
	--error_margin=3
