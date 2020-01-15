#!/usr/bin/env sh

./build/tools/caffe test \
	--model=models/ZynqNet/RistrettoDemo/quantized_dfixed.prototxt \
	--weights=models/ZynqNet/zynqnet.caffemodel \
	--gpu=0 --iterations=2000
