#!/usr/bin/env sh

./build/tools/caffe train \
	--solver=models/SqueezeNet_v1.1/RistrettoDemo/solver_finetune.prototxt \
	--weights=models/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
