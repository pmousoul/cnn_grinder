#! /bin/bash

wget https://raw.githubusercontent.com/dgschwend/zynqnet/master/_TRAINED_MODEL/deploy.prototxt
wget https://github.com/dgschwend/zynqnet/raw/master/_TRAINED_MODEL/snapshot_iter_300280.caffemodel
mv snapshot_iter_300280.caffemodel zynqnet.caffemodel
wget https://raw.githubusercontent.com/dgschwend/zynqnet/master/_TRAINED_MODEL/train_val.prototxt

echo "****************************************************************************"
echo "*Don't forget to change the dataset source paths for training and testing!!*" 
echo "****************************************************************************"

