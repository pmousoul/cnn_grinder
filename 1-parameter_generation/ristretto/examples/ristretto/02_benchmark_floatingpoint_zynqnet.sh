#!/usr/bin/env sh

./build/tools/caffe test \
	--model=models/ZynqNet/train_val.prototxt \
	--weights=models/ZynqNet/zynqnet.caffemodel \
	--gpu=0 --iterations=2000
