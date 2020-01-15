% Preprocess image according to caffe instructions:
% http://caffe.berkeleyvision.org/tutorial/interfaces.html

function [preproc_img] = image_preproc(img_file, meanFile, CROPPED_DIM)

img = imread(img_file);

% If we have one channel image then copy it to 3 channels
if (size(img, 3) == 1)
    img1(:, :, 1) = img;
    img1(:, :, 2) = img;
    img1(:, :, 3) = img;
    img = img1;
end

% permute channels from RGB to BGR
img = img(:, :, [3, 2, 1]);
% flip width and height to make width the fastest dimension
img = permute(img, [2, 1, 3]);
% convert from uint8 to single
img = single(img);

% % load dataset mean file
% img_mean = load(meanFile);
% name = fieldnames(img_mean);
% datasetMean = img_mean.(name{1});

% % resize input image to imagenet mean image file size and
% % subtract the imagenet mean image values from the input image
% sz = size(datasetMean);
% img = imresize(img, [sz(1) sz(2)], 'bilinear','AntiAliasing',false);
%     % img = imresize(img, [sz(1) sz(2)], 'bilinear');
% img = img - datasetMean;

% we don't use the mean file
% we just subtract mean values from each image channel
img(:, :, 1) = img(:, :, 1) - 104;
img(:, :, 2) = img(:, :, 2) - 117;
img(:, :, 3) = img(:, :, 3) - 123;

% reshape to a fixed size (e.g., 227x227).
img = imresize(img, [CROPPED_DIM CROPPED_DIM], 'bilinear','AntiAliasing',false);
    % img = imresize(img, [CROPPED_DIM CROPPED_DIM], 'bilinear');
preproc_img = zeros(CROPPED_DIM, CROPPED_DIM, 3, 'double');
preproc_img(:, :, :) = img;

end