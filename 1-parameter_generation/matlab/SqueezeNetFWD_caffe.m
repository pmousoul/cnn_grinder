% Forward path implementation of SqueezeNet v1.1 - Caffe version.
clc;
clear all;
restoredefaultpath;


% Load image file, dataset mean file, compare inter-layer results option,
% add matcaffe PATH, and binary file generation option
[inputFile, meanFile, cmp, matcaffePath, bin] = read_config('config.txt');


% Add/set paths
addpath( matcaffePath );
addpath( './Extract_Params/sqn/' );
addpath( './Layers/' );
model_path = './Matcaffe/sqn/';
param_path = './Parameters/caffe/sqn/';
ires_path = './Inter-layer_Results/caffe/sqn/';


% Preprocess input image
CROPPED_DIM = 227;
img = image_preproc(inputFile, meanFile, CROPPED_DIM);


% Extract network parameters ( if (extract_params==1) ).
% Extract intermediate results (.mat for Matlab)
% for the image specified
extract_params=1;
extract_iresults=cmp;
extract_params_iresults_caffe(extract_params, extract_iresults, {img}, model_path, param_path, ires_path);


% Convolution Layer 1
load([param_path 'conv1_w.mat']); load([param_path 'conv1_b.mat']);
conv_rslt = layer_conv_caffe(img, weights, bias, 3, 2, 0);
conv_rslt = layer_relu(conv_rslt);
if (cmp)
    fprintf('\n');
    fprintf('SqueezeNet v1.1\n');
    fprintf('Caffe version\n');
    fprintf('K x K x N x M (matlab) parameter organization\n');
    fprintf('Check intermediate results\n');
    fprintf('against Caffe (floating point)\n');
    fprintf('using floating-point input\n');
    fprintf('and parameter accuracy: \n\n');
    %
    load([ires_path '2_conv1.mat']);
    fprintf('Max error in conv1: \t%f\n', max(abs(data(:) - conv_rslt(:))));
end


% Pooling Layer 1
pool_rslt = layer_maxpool_caffe(conv_rslt, 3, 2, 0);
if (cmp)
    load([ires_path '3_pool1.mat']);
    fprintf('Max error in maxpool1: \t%f\n', max(abs(data(:) - pool_rslt(:))));
end


% Fire Layer 2
load([param_path 'fire2_squeeze1x1_w.mat']); load([param_path 'fire2_squeeze1x1_b.mat']);
conv_rslt = layer_conv_caffe(pool_rslt, weights, bias, 1, 1, 0);
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire2_expand1x1_w.mat']); load([param_path 'fire2_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire2_expand3x3_w.mat']); load([param_path 'fire2_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_caffe(conv_rslt, weights, bias, 3, 1, 1);
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (56, 56, 128);
conv_rslt (:, :, 1:64) = conv_rslt_1; conv_rslt (:, :, 65:128) = conv_rslt_2;

if (cmp)
    load([ires_path '4_fire2.mat']);
    fprintf('Max error in Fire2: \t%f\n', max(abs(data(:) - conv_rslt(:))));
end


% Fire Layer 3
load([param_path 'fire3_squeeze1x1_w.mat']); load([param_path 'fire3_squeeze1x1_b.mat']);
conv_rslt = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire3_expand1x1_w.mat']); load([param_path 'fire3_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire3_expand3x3_w.mat']); load([param_path 'fire3_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_caffe(conv_rslt, weights, bias, 3, 1, 1);
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (56, 56, 128);
conv_rslt (:, :, 1:64) = conv_rslt_1; conv_rslt (:, :, 65:128) = conv_rslt_2;

if (cmp)
    load([ires_path '5_fire3.mat']);
    fprintf('Max error in Fire3: \t%f\n', max(abs(data(:) - conv_rslt(:))));
end


% Pooling Layer 3
pool_rslt = layer_maxpool_caffe(conv_rslt, 3, 2, 1);

if (cmp)
    load([ires_path '6_pool3.mat']);
    fprintf('Max error in maxpool3: \t%f\n', max(abs(data(:) - pool_rslt(:))));
end


% Fire Layer 4
load([param_path 'fire4_squeeze1x1_w.mat']); load([param_path 'fire4_squeeze1x1_b.mat']);
conv_rslt = layer_conv_caffe(pool_rslt, weights, bias, 1, 1, 0);
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire4_expand1x1_w.mat']); load([param_path 'fire4_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire4_expand3x3_w.mat']); load([param_path 'fire4_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_caffe(conv_rslt, weights, bias, 3, 1, 1);
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (28, 28, 256);
conv_rslt (:, :, 1:128) = conv_rslt_1; conv_rslt (:, :, 129:256) = conv_rslt_2;

if (cmp)
    load([ires_path '7_fire4.mat']);
    fprintf('Max error in Fire4: \t%f\n', max(abs(data(:) - conv_rslt(:))));
end


% Fire Layer 5
load([param_path 'fire5_squeeze1x1_w.mat']); load([param_path 'fire5_squeeze1x1_b.mat']);
conv_rslt = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire5_expand1x1_w.mat']); load([param_path 'fire5_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire5_expand3x3_w.mat']); load([param_path 'fire5_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_caffe(conv_rslt, weights, bias, 3, 1, 1);
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (28, 28, 256);
conv_rslt (:, :, 1:128) = conv_rslt_1; conv_rslt (:, :, 129:256) = conv_rslt_2;

if (cmp)
    load([ires_path '8_fire5.mat']);
    fprintf('Max error in Fire5: \t%f\n', max(abs(data(:) - conv_rslt(:))));
end


% Pooling Layer 5
pool_rslt = layer_maxpool_caffe(conv_rslt, 3, 2, 1);

if (cmp)
    load([ires_path '9_pool5.mat']);
    fprintf('Max error in maxpool5: \t%f\n', max(abs(data(:) - pool_rslt(:))));
end


% Fire Layer 6
load([param_path 'fire6_squeeze1x1_w.mat']); load([param_path 'fire6_squeeze1x1_b.mat']);
conv_rslt = layer_conv_caffe(pool_rslt, weights, bias, 1, 1, 0);
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire6_expand1x1_w.mat']); load([param_path 'fire6_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire6_expand3x3_w.mat']); load([param_path 'fire6_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_caffe(conv_rslt, weights, bias, 3, 1, 1);
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (14, 14, 384);
conv_rslt (:, :, 1:192) = conv_rslt_1; conv_rslt (:, :, 193:384) = conv_rslt_2;

if (cmp)
    load([ires_path '10_fire6.mat']);
    fprintf('Max error in Fire6: \t%f\n', max(abs(data(:) - conv_rslt(:))));
end


% Fire Layer 7
load([param_path 'fire7_squeeze1x1_w.mat']); load([param_path 'fire7_squeeze1x1_b.mat']);
conv_rslt = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire7_expand1x1_w.mat']); load([param_path 'fire7_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire7_expand3x3_w.mat']); load([param_path 'fire7_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_caffe(conv_rslt, weights, bias, 3, 1, 1);
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (14, 14, 384);
conv_rslt (:, :, 1:192) = conv_rslt_1; conv_rslt (:, :, 193:384) = conv_rslt_2;

if (cmp)
    load([ires_path '11_fire7.mat']);
    fprintf('Max error in Fire7: \t%f\n', max(abs(data(:) - conv_rslt(:))));
end


% Fire Layer 8
load([param_path 'fire8_squeeze1x1_w.mat']); load([param_path 'fire8_squeeze1x1_b.mat']);
conv_rslt = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire8_expand1x1_w.mat']); load([param_path 'fire8_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire8_expand3x3_w.mat']); load([param_path 'fire8_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_caffe(conv_rslt, weights, bias, 3, 1, 1);
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (14, 14, 512);
conv_rslt (:, :, 1:256) = conv_rslt_1; conv_rslt (:, :, 257:512) = conv_rslt_2;

if (cmp)
    load([ires_path '12_fire8.mat']);
    fprintf('Max error in Fire8: \t%f\n', max(abs(data(:) - conv_rslt(:))));
end


% Fire Layer 9
load([param_path 'fire9_squeeze1x1_w.mat']); load([param_path 'fire9_squeeze1x1_b.mat']);
conv_rslt = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt = layer_relu(conv_rslt);

load([param_path 'fire9_expand1x1_w.mat']); load([param_path 'fire9_expand1x1_b.mat']);
conv_rslt_1 = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt_1 = layer_relu(conv_rslt_1);

load([param_path 'fire9_expand3x3_w.mat']); load([param_path 'fire9_expand3x3_b.mat']);
conv_rslt_2 = layer_conv_caffe(conv_rslt, weights, bias, 3, 1, 1);
conv_rslt_2 = layer_relu(conv_rslt_2);

conv_rslt = zeros (14, 14, 512);
conv_rslt (:, :, 1:256) = conv_rslt_1; conv_rslt (:, :, 257:512) = conv_rslt_2;

if (cmp)
    load([ires_path '13_fire9.mat']);
    fprintf('Max error in Fire9: \t%f\n', max(abs(data(:) - conv_rslt(:))));
end


% Convolution Layer 10
load([param_path 'conv10_w.mat']); load([param_path 'conv10_b.mat']);
conv_rslt = layer_conv_caffe(conv_rslt, weights, bias, 1, 1, 0);
conv_rslt = layer_relu(conv_rslt);

if (cmp)
    load([ires_path '14_conv10.mat']);
    fprintf('Max error in conv10: \t%f\n', max(abs(data(:) - conv_rslt(:))));
end


% Average Pooling Layer
pool_rslt = layer_avgpool_caffe(conv_rslt, 14, 1);

if (cmp)
    load([ires_path '15_pool10.mat']);
    fprintf('Max error in avgpool10: %f\n', max(abs(data(:) - pool_rslt(:))));
end


% Softmax
soft_rslt = layer_softmax(pool_rslt);

if (cmp)
    load([ires_path '16_prob.mat']);
    fprintf('Max error in softmax: \t%f\n', max(abs(data(:) - soft_rslt(:))));
end


% Print the inference result
load('Labels/class_labels.mat');
[sorted_labels,sorted_indices] = sort(soft_rslt,'descend');
fprintf('\nTop-5 results =\n');
for i = 1:5
    fprintf('%f %s\n', sorted_labels(i), class_labels{sorted_indices(i)});
end