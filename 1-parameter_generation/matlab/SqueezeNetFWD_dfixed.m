% Forward path implementation of SqueezeNet v1.1 - Dynamic fixed-point version
clc;
clear all;
restoredefaultpath;


% Load image file, dataset mean file, compare inter-layer results option,
% add matcaffe PATH, and binary file generation option
[inputFile, meanFile, cmp, matcaffePath, bin] = read_config('config.txt');


% Add/set paths
addpath( matcaffePath );
addpath( './Extract_Params/sqn/' );
addpath( './Params_To_Binary_Files/' );
addpath( './Params_To_Binary_Files/sqn' );
addpath( './Layers/' );
model_path = './Matcaffe/sqn/';
param_path = './Parameters/dfixed/sqn/';
ires_path = './Inter-layer_Results/dfixed/sqn/';
bin_path = './Binary_Files/dfixed/sqn/';


% Width and fractional part of parameters (w,f) and feature maps (tw, tf)

% This is what Ristretto gives:
% tfi=[ 0,   -2, -4, -4,   -3, -3, -3,   -3, -4, -4,   -4, -4, -4,   -3, -4, -4,   -3, -4, -4,   -3, -3, -3,   -3, -3, -3,   -3];
% tfo=[-2,   -4, -3, -3,   -3, -3, -3,   -4, -3, -4,   -4, -3, -3,   -4, -3, -3,   -4, -2, -3,   -3, -2, -3,   -3, -2, -2,   -2];
% f  =[ 7,    6,  7,  7,    7,  7,  7,    6,  7,  7,    7,  7,  7,    7,  8,  7,    7,  8,  8,    7,  7,  7,    8,  7,  8,    8];

twi=  8;
two=  8;
w  =  8;
tfi=[ 0,   -3, -4, -4,   -3, -3, -3,   -3, -4, -4,   -4, -4, -4,   -3, -4, -4,   -3, -4, -4,   -3, -3, -3,   -3, -3, -3,   -2];
tfo=[-3,   -4, -3, -3,   -3, -3, -3,   -4, -4, -4,   -4, -3, -3,   -4, -3, -3,   -4, -3, -3,   -3, -3, -3,   -3, -2, -2,   -1];
f  =[ 7,    6,  7,  7,    7,  7,  7,    6,  7,  7,    7,  7,  7,    7,  8,  7,    7,  8,  8,    7,  7,  7,    8,  7,  8,    8];


% Preprocess input image
CROPPED_DIM = 227; % Image crop size
img = image_preproc(inputFile, meanFile, CROPPED_DIM);
% Convert input image to int8
img = fi(img, true, twi, tfi(1));
img = storedInteger(img);


% Extract network parameters ( if (extract_params==1) ).
% Extract intermediate results for the image specified
% for both the Matlab and the C++ implementation
extract_params=1;
extract_iresults=cmp;
% Weights-bias fixed-point value parameters
is_fixed=1;
%
extract_params_iresults( ...
    extract_params, extract_iresults, w, f, is_fixed, img, ...
    model_path, 'deploy.prototxt', 'quantized_dfixed.prototxt', param_path, ires_path, bin_path);


% Write input file and network parameters to binary files
% for use in the C/C++ implementation
% Value type of saved data
value_type_p = 'int8'; % parameters
value_type_fm = 'int8'; % feature maps
% Parameter and image file paths
param_file = [bin_path 'params_dfixed.bin'];
image_file = [bin_path 'image_dfixed.bin'];
%
if (bin==1)
    mat_to_bin(param_path, param_file, value_type_p, img, image_file, value_type_fm );
end


% Permute the input image dimensions from W x H x N to N x W x H
img = permute(img,[3 1 2]);


% Convolution Layer 1
load([param_path 'conv1_w.mat']); load([param_path 'conv1_b.mat']);

% Reshape
reshape_rslt = layer_reshape( img, 3, 2, 32, 0, 'int8' );
conv_rslt = layer_conv_dfixed(reshape_rslt, weights, bias, 1, 1, 0, tfi(1), tfo(1), f(1), 'int8' );

conv_rslt = layer_relu(conv_rslt);
if (cmp)
    fprintf('\n');
    fprintf('SqueezeNet v1.1\n');
    fprintf('Dynamic Fixed Point - DFP version\n');
    fprintf('N x K x K x M (matlab) parameter organization\n');
    fprintf('Check intermediate results\n');
    fprintf('against Ristretto (dynamic fixed point)\n');
    fprintf('using 8-bit input and parameters accuracy: \n\n');

    load([ires_path '2_conv1.mat']);
    fprintf('MAX error in conv1: %f\n', max( double( data(:) ) - ( double( conv_rslt(:) ) * ( 2^(-tfo(1)) ) ) ));
end


% Pooling Layer 1
pool_rslt = layer_maxpool(conv_rslt, 3, 2, 0, 'int8');

if (cmp)
    load([ires_path '3_pool1.mat']);
    fprintf('MAX error in maxpool1: %f\n', max( double( data(:) ) - ( double( pool_rslt(:) ) * ( 2^(-tfo(1)) ) ) ));
end


% Fire Layer 2
load([param_path 'fire2_squeeze1x1_w.mat']); load([param_path 'fire2_squeeze1x1_b.mat']);
conv_rslt = layer_conv_dfixed(pool_rslt, weights, bias, 1, 1, 0, tfi(2), tfo(2), f(2), 'int8' );
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire2_expand1x1_w.mat']); load([param_path 'fire2_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(3), tfo(3), f(3), 'int8' );
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire2_expand3x3_w.mat']); load([param_path 'fire2_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_dfixed(conv_rslt, weights, bias, 3, 1, 1, tfi(4), tfo(4), f(4), 'int8' );
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (128, 56, 56, 'int8');
conv_rslt (1:64, :, :) = conv_rslt_1; conv_rslt (65:128, :, :) = conv_rslt_2;

if (cmp)
    load([ires_path '4_fire2.mat']);
    fprintf('MAX error in Fire2: %f\n', max( double( data(:) ) - ( double( conv_rslt(:) ) * ( 2^(-tfo(4)) ) ) ));
end


% Fire Layer 3
load([param_path 'fire3_squeeze1x1_w.mat']); load([param_path 'fire3_squeeze1x1_b.mat']);
conv_rslt = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(5), tfo(5), f(5), 'int8' );
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire3_expand1x1_w.mat']); load([param_path 'fire3_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(6), tfo(6), f(6), 'int8' );
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire3_expand3x3_w.mat']); load([param_path 'fire3_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_dfixed(conv_rslt, weights, bias, 3, 1, 1, tfi(7), tfo(7), f(7), 'int8' );
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (128, 56, 56, 'int8');
conv_rslt (1:64, :, :) = conv_rslt_1; conv_rslt (65:128, :, :) = conv_rslt_2;

if (cmp)
    load([ires_path '5_fire3.mat']);
    fprintf('MAX error in Fire3: %f\n', max( double( data(:) ) - ( double( conv_rslt(:) ) * ( 2^(-tfo(7)) ) ) ));
end


% Pooling Layer 3
pool_rslt = layer_maxpool(conv_rslt, 3, 2, 1, 'int8');

if (cmp)
    load([ires_path '6_pool3.mat']);
    fprintf('MAX error in maxpool3: %f\n', max( double( data(:) ) - ( double( pool_rslt(:) ) * ( 2^(-tfo(7)) ) ) ));
end


% Fire Layer 4
load([param_path 'fire4_squeeze1x1_w.mat']); load([param_path 'fire4_squeeze1x1_b.mat']);
conv_rslt = layer_conv_dfixed(pool_rslt, weights, bias, 1, 1, 0, tfi(8), tfo(8), f(8), 'int8' );
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire4_expand1x1_w.mat']); load([param_path 'fire4_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(9), tfo(9), f(9), 'int8' );
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire4_expand3x3_w.mat']); load([param_path 'fire4_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_dfixed(conv_rslt, weights, bias, 3, 1, 1, tfi(10), tfo(10), f(10), 'int8' );
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (256, 28, 28, 'int8');
conv_rslt (1:128, :, :) = conv_rslt_1; conv_rslt (129:256, :, :) = conv_rslt_2;

if (cmp)
    load([ires_path '7_fire4.mat']);
    fprintf('MAX error in Fire4: %f\n', max( double( data(:) ) - ( double( conv_rslt(:) ) * ( 2^(-tfo(10)) ) ) ));
end


% Fire Layer 5
load([param_path 'fire5_squeeze1x1_w.mat']); load([param_path 'fire5_squeeze1x1_b.mat']);
conv_rslt = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(11), tfo(11), f(11), 'int8' );
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire5_expand1x1_w.mat']); load([param_path 'fire5_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(12), tfo(12), f(12), 'int8' );
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire5_expand3x3_w.mat']); load([param_path 'fire5_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_dfixed(conv_rslt, weights, bias, 3, 1, 1, tfi(13), tfo(13), f(13), 'int8' );
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (256, 28, 28, 'int8');
conv_rslt (1:128, :, :) = conv_rslt_1; conv_rslt (129:256, :, :) = conv_rslt_2;

if (cmp)
    load([ires_path '8_fire5.mat']);
    fprintf('MAX error in Fire5: %f\n', max( double( data(:) ) - ( double( conv_rslt(:) ) * ( 2^(-tfo(13)) ) ) ));
end


% Pooling Layer 5
pool_rslt = layer_maxpool(conv_rslt, 3, 2, 1, 'int8');

if (cmp)
    load([ires_path '9_pool5.mat']);
    fprintf('MAX error in maxpool5: %f\n', max( double( data(:) ) - ( double( pool_rslt(:) ) * ( 2^(-tfo(13)) ) ) ));
end


% Fire Layer 6
load([param_path 'fire6_squeeze1x1_w.mat']); load([param_path 'fire6_squeeze1x1_b.mat']);
conv_rslt = layer_conv_dfixed(pool_rslt, weights, bias, 1, 1, 0, tfi(14), tfo(14), f(14), 'int8' );
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire6_expand1x1_w.mat']); load([param_path 'fire6_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(15), tfo(15), f(15), 'int8' );
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire6_expand3x3_w.mat']); load([param_path 'fire6_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_dfixed(conv_rslt, weights, bias, 3, 1, 1, tfi(16), tfo(16), f(16), 'int8' );
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (384, 14, 14, 'int8');
conv_rslt (1:192, :, :) = conv_rslt_1; conv_rslt (193:384, :, :) = conv_rslt_2;

if (cmp)
    load([ires_path '10_fire6.mat']);
    fprintf('MAX error in Fire6: %f\n', max( double( data(:) ) - ( double( conv_rslt(:) ) * ( 2^(-tfo(16)) ) ) ));
end


% Fire Layer 7
load([param_path 'fire7_squeeze1x1_w.mat']); load([param_path 'fire7_squeeze1x1_b.mat']);
conv_rslt = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(17), tfo(17), f(17), 'int8' );
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire7_expand1x1_w.mat']); load([param_path 'fire7_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(18), tfo(18), f(18), 'int8' );
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire7_expand3x3_w.mat']); load([param_path 'fire7_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_dfixed(conv_rslt, weights, bias, 3, 1, 1, tfi(19), tfo(19), f(19), 'int8' );
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (384, 14, 14, 'int8');
conv_rslt (1:192, :, :) = conv_rslt_1; conv_rslt (193:384, :, :) = conv_rslt_2;

if (cmp)
    load([ires_path '11_fire7.mat']);
    fprintf('MAX error in Fire7: %f\n', max( double( data(:) ) - ( double( conv_rslt(:) ) * ( 2^(-tfo(19)) ) ) ));
end


% Fire Layer 8
load([param_path 'fire8_squeeze1x1_w.mat']); load([param_path 'fire8_squeeze1x1_b.mat']);
conv_rslt = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(20), tfo(20), f(20), 'int8' );
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire8_expand1x1_w.mat']); load([param_path 'fire8_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(21), tfo(21), f(21), 'int8' );
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire8_expand3x3_w.mat']); load([param_path 'fire8_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_dfixed(conv_rslt, weights, bias, 3, 1, 1, tfi(22), tfo(22), f(22), 'int8' );
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (512, 14, 14, 'int8');
conv_rslt (1:256, :, :) = conv_rslt_1; conv_rslt (257:512, :, :) = conv_rslt_2;

if (cmp)
    load([ires_path '12_fire8.mat']);
    fprintf('MAX error in Fire8: %f\n', max( double( data(:) ) - ( double( conv_rslt(:) ) * ( 2^(-tfo(22)) ) ) ));
end


% Fire Layer 9
load([param_path 'fire9_squeeze1x1_w.mat']); load([param_path 'fire9_squeeze1x1_b.mat']);
conv_rslt = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(23), tfo(23), f(23), 'int8' );
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire9_expand1x1_w.mat']); load([param_path 'fire9_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(24), tfo(24), f(24), 'int8' );
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire9_expand3x3_w.mat']); load([param_path 'fire9_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_dfixed(conv_rslt, weights, bias, 3, 1, 1, tfi(25), tfo(25), f(25), 'int8' );
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (512, 14, 14, 'int8');
conv_rslt (1:256, :, :) = conv_rslt_1; conv_rslt (257:512, :, :) = conv_rslt_2;

if (cmp)
    load([ires_path '13_fire9.mat']);
    fprintf('MAX error in Fire9: %f\n', max( double( data(:) ) - ( double( conv_rslt(:) ) * ( 2^(-tfo(25)) ) ) ));
end


% Convolution Layer 10
load([param_path 'conv10_w.mat']); load([param_path 'conv10_b.mat']);
conv_rslt = layer_conv_dfixed(conv_rslt, weights, bias, 1, 1, 0, tfi(26), tfo(26), f(26), 'int8' );
conv_rslt = layer_relu(conv_rslt);

if (cmp)
    load([ires_path '14_conv10.mat']);
    fprintf('MAX error in conv10: %f\n', max( double( data(:) ) - ( double( conv_rslt(:) ) * ( 2^(-tfo(26)) ) ) ));
end


% Convert to double
conv_rslt = double( conv_rslt ) * 2^(-tfo(26));


% Average Pooling Layer
pool_rslt = layer_avgpool(conv_rslt, 14, 1);
if (cmp)
    load([ires_path '15_pool10.mat']);
    fprintf('MAX error in avgpool10: %f\n', max( double( data(:) ) - double( pool_rslt(:) ) ) );
end


% Softmax
soft_rslt = layer_softmax(pool_rslt);
load([ires_path '16_prob.mat']);
if (cmp)
    fprintf('MAX error in softmax: %f\n', max( double( data(:) ) - double( soft_rslt(:) ) ) ) ;
end


% Print the inference result
load('Labels/class_labels.mat');
[sorted_probs,sorted_indices] = sort( soft_rslt,'descend');
fprintf('\nTop-5 results =\n');
for i = 1:5
    fprintf('%f %s\n', sorted_probs(i), class_labels{ uint64(sorted_indices(i))});
end
