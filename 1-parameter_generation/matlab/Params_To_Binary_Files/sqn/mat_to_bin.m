% Convert mat parameter files and preprocessed image to binary files for
% use in C/C++ code - float/fixed-point version

function [] = mat_to_bin(param_path, param_file, value_type_p, img, image_file, value_type_fm )


% conv1
load([param_path 'conv1_b.mat'])
load([param_path 'conv1_w.mat'])
params_to_bin(weights, bias, param_file, 'w', value_type_p);

% fire2
load([param_path 'fire2_squeeze1x1_b.mat'])
load([param_path 'fire2_squeeze1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire2_expand1x1_b.mat'])
load([param_path 'fire2_expand1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire2_expand3x3_b.mat'])
load([param_path 'fire2_expand3x3_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);

% fire3
load([param_path 'fire3_squeeze1x1_b.mat'])
load([param_path 'fire3_squeeze1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire3_expand1x1_b.mat'])
load([param_path 'fire3_expand1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire3_expand3x3_b.mat'])
load([param_path 'fire3_expand3x3_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);

% fire4
load([param_path 'fire4_squeeze1x1_b.mat'])
load([param_path 'fire4_squeeze1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire4_expand1x1_b.mat'])
load([param_path 'fire4_expand1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire4_expand3x3_b.mat'])
load([param_path 'fire4_expand3x3_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);

% fire5
load([param_path 'fire5_squeeze1x1_b.mat'])
load([param_path 'fire5_squeeze1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire5_expand1x1_b.mat'])
load([param_path 'fire5_expand1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire5_expand3x3_b.mat'])
load([param_path 'fire5_expand3x3_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);

% fire6
load([param_path 'fire6_squeeze1x1_b.mat'])
load([param_path 'fire6_squeeze1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire6_expand1x1_b.mat'])
load([param_path 'fire6_expand1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire6_expand3x3_b.mat'])
load([param_path 'fire6_expand3x3_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);

% fire7
load([param_path 'fire7_squeeze1x1_b.mat'])
load([param_path 'fire7_squeeze1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire7_expand1x1_b.mat'])
load([param_path 'fire7_expand1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire7_expand3x3_b.mat'])
load([param_path 'fire7_expand3x3_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);

% fire8
load([param_path 'fire8_squeeze1x1_b.mat'])
load([param_path 'fire8_squeeze1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire8_expand1x1_b.mat'])
load([param_path 'fire8_expand1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire8_expand3x3_b.mat'])
load([param_path 'fire8_expand3x3_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);

% fire9
load([param_path 'fire9_squeeze1x1_b.mat'])
load([param_path 'fire9_squeeze1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire9_expand1x1_b.mat'])
load([param_path 'fire9_expand1x1_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);
%
load([param_path 'fire9_expand3x3_b.mat'])
load([param_path 'fire9_expand3x3_w.mat'])
params_to_bin(weights, bias, param_file, 'a', value_type_p);

% conv10
load([param_path 'conv10_b.mat'])
load([param_path 'conv10_w.mat'])
% we want weights to have at most 256 output channels
% if there are more than 256 output channels, we need to
% partition this dimension
weights1 = weights(:,:,:,1:256);
bias1 = bias(1:256);
weights2 = weights(:,:,:,257:512);
bias2 = bias(257:512);
weights3 = weights(:,:,:,513:768);
bias3 = bias(513:768);
% output channels must be a multiple of 16 so we fill the
% last partition with zeros
s = size(weights);
weights4 = zeros( s(1), s(2), s(3), 256, 'like', weights);
weights4(:,:,:,1:232) = weights(:,:,:,769:1000);
bias4 = zeros( 256, 'like', bias);
bias4(1:232) = bias(769:1000);
%
params_to_bin(weights1, bias1, param_file, 'a', value_type_p);
params_to_bin(weights2, bias2, param_file, 'a', value_type_p);
params_to_bin(weights3, bias3, param_file, 'a', value_type_p);
params_to_bin(weights4, bias4, param_file, 'a', value_type_p);

% Convert preprocessed image to binary file
% Permute the input image dimensions from W x H x N to N x W x H
img = permute(img,[3 1 2]);
% convert to row-major and save it to binary file
var_to_bin(img, image_file, 'w', value_type_fm);

end