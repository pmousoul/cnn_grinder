#!/usr/bin/env sh

./build/tools/caffe test \
	--model=models/SqueezeNet_v1.1/RistrettoDemo/quantized_dfixed.prototxt \
	--weights=models/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel \
	--gpu=0 --iterations=2000
