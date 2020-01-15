#! /bin/bash

wget https://github.com/DeepScale/SqueezeNet/raw/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
wget https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/deploy.prototxt
wget https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/train_val.prototxt

echo "****************************************************************************"
echo "*Don't forget to change the dataset source paths for training and testing!!*" 
echo "****************************************************************************"

