# Matlab `1-parameter generation` project description

## About

This project is an adaptation and an extension of the following project: https://github.com/mtmd/SqueezeNet_MATLAB
We extent the above work to support the forward pass of SqueezeNet v1.1 and ZynqNet in floating-point and (dynamic) fixed-point modes. Matcaffe, a Caffe Matlab interface, is used to run Caffe inside Matlab and generate the network parameters and inter-layer network data in order to compare the Caffe results against those of the Matlab custom algorithmic implementation. Additionally, Ristretto, a Caffe extension, is used to reduce the parameter bit-width (quantization), without hurting the network accuracy. The results from Ristretto are used in the Matlab implementation to generate the inter-layer results and the network parameters for a (dynamic) fixed-point network. The inter-layer results are used to verify that the custom (dynamic) fixed-point Matlab implementation is as accurate as Ristretto. Additionally, there is a Matlab script that can be used to save the generated network parameters and input image to binary files. These binary files can be used in the HLS C/C++ implementation of the network's forward pass. The HLS C/C++ implementation is used for synthesizing and implementing an FPGA SoC design.


## Prerequisites

Since Ristretto is a Caffe extension (and it is built on top of Caffe), the only requirements to run this project are the installation of Ristretto and the download/configuration of the ImageNet 2012 (ILSVRC12) dataset. The following steps were tested on an Ubuntu 16.04 machine, equiped with a CUDA capable GPU (a GPU is mandatory to accelerate Ristretto), and running Matlab 2016b:

1. Clone the Ristretto repository:
    git clone https://github.com/pmgysel/caffe.git
2. Follow the Caffe installation instructions and build Ristretto (and thus Caffe) according to your OS (be ready for surprises - internet is your friend). Use the following instructions and do not forget to build Matcaffe using 'make matcaffe':
    http://caffe.berkeleyvision.org/installation.html
    (https://github.com/BVLC/caffe/issues/4808#issuecomment-251427053)
3. Assuming that you have downloaded the ILSVRC12 dataset, use the following steps to prepare the dataset for Ristretto:
    http://caffe.berkeleyvision.org/gathered/examples/imagenet.html


## How to run

1. Make sure to set all the required parameters in the `config.txt` file:
	`input_file`	refers to the input image file location
	`cmp`			enables to compare intermediate results with the floating-point/(dynamic) fixed-point results (use `1` for enable and `0` for disable)
    `matcaffe_path` refers to the matcaffe path
    `bin`			creates parameter and image binary files (use `1` for enable and `0` for disable)

3. Make sure to download the SqueezeNet v1.1 files and the ZynqNet files into the `Matcaffe/sqn` and the `Matcaffe/zqn` folders respectively from the following repos:
https://github.com/DeepScale/SqueezeNet
https://github.com/dgschwend/zynqnet/tree/master/_TRAINED_MODEL
(use the download_model_files.sh scripts)
Make sure that the model/parameter file names match to those used in the scripts found in the `Extract_Params` folder tree.
You should also edit the .prototxt files to be usable by the Matlab scripts - (see for example the `deploy.prototxt` and the `train_val.prototxt` files in the `1-parameter_generation/matlab/Matcaffe/sqn/` directory; compare the beggining and the last layer descriptions in these two files).

4. Using Matlab, run `SqueezeNetFWD_float.m` (or the `SqueezeNetFWD_dfixed.m` which refers to the dynamic fixed-point implementation) to check that everything runs as expected.

5. After running `SqueezeNetFWD_dfixed.m`, dynamic fixed-point CNN parameters are generated in the `Parameters/dfixed/sqn` folder and inter-layer results are generated in the `Inter-layer_Results/dfixed/sqn` folder. The parameters and the inter-layer results are saved in binary files (in row major format) for use in custom C/C++ or HLS (FPGA) implementations in the `Binary_Files/dfixed/sqn` folder. Ristretto is used for setting the accuracy of the dynamic fixed-point parameters in the `SqueezeNetFWD_dfixed.m` file.
