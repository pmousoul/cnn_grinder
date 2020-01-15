% Extract layer parameters and intermediate network results
% from SqueezeNet v1.1 CNN using matcaffe

function extract_params_iresults( ...
    extract_params, extract_iresults, w, f, is_fixed, img, ...
    model_path, model_name, q_model_name, param_path, ires_path, bin_path )


% Set caffe mode
%caffe.set_mode_cpu();


% Initialize the network using the squeezenet_v1.1.caffemodel
net_model  = [model_path model_name];
net_params = [model_path 'squeezenet_v1.1.caffemodel'];
phase      = 'test'; % run with phase test (so that dropout isn't applied)


% Initialize the network
net = caffe.Net(net_model, net_params, phase);


% Get the network parameters and save them to .mat and .bin
% for use in the Matlab and the C/C++ implementation respectively
if ( extract_params==1 )
    %conv1
    weights = net.params('conv1',1).get_data();
    %net.params('conv1', 1).set_data(weights);
    % permute form K x K x N x M to N x K x K x M
    weights = permute(weights, [ 3 1 2 4 ]);

    % Reshape
    tmp = reshape(weights, [27, 1, 1, 64]);
    weights = zeros(32,1,1,64, 'like', tmp);
    weights(1:27,:,:,:) = tmp;
    
    if (is_fixed==1)
        weights = fi(weights, true, w, f(1));
        weights = storedInteger(weights);
    end
    save( [ param_path 'conv1_w.mat' ],  'weights');
    %
    bias = net.params('conv1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(1));
        bias = storedInteger(bias);
    end
    save( [ param_path 'conv1_b.mat' ],  'bias');
    %
    %fire2
    weights = net.params('fire2/squeeze1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(2));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire2_squeeze1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire2/squeeze1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(2));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire2_squeeze1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire2/expand1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(3));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire2_expand1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire2/expand1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(3));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire2_expand1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire2/expand3x3',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(4));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire2_expand3x3_w.mat' ],  'weights');
    %
    bias = net.params('fire2/expand3x3',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(4));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire2_expand3x3_b.mat' ],  'bias');
    %
    %fire3
    weights = net.params('fire3/squeeze1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(5));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire3_squeeze1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire3/squeeze1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(5));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire3_squeeze1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire3/expand1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(6));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire3_expand1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire3/expand1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(6));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire3_expand1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire3/expand3x3',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(7));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire3_expand3x3_w.mat' ],  'weights');
    %
    bias = net.params('fire3/expand3x3',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(7));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire3_expand3x3_b.mat' ],  'bias');
    %
    %fire4
    weights = net.params('fire4/squeeze1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(8));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire4_squeeze1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire4/squeeze1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(8));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire4_squeeze1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire4/expand1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(9));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire4_expand1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire4/expand1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(9));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire4_expand1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire4/expand3x3',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(10));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire4_expand3x3_w.mat' ],  'weights');
    %
    bias = net.params('fire4/expand3x3',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(10));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire4_expand3x3_b.mat' ],  'bias');
    %
    %fire5
    weights = net.params('fire5/squeeze1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(11));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire5_squeeze1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire5/squeeze1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(11));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire5_squeeze1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire5/expand1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(12));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire5_expand1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire5/expand1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(12));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire5_expand1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire5/expand3x3',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(13));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire5_expand3x3_w.mat' ],  'weights');
    %
    bias = net.params('fire5/expand3x3',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(13));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire5_expand3x3_b.mat' ],  'bias');
    %
    %fire6
    weights = net.params('fire6/squeeze1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(14));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire6_squeeze1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire6/squeeze1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(14));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire6_squeeze1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire6/expand1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(15));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire6_expand1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire6/expand1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(15));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire6_expand1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire6/expand3x3',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(16));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire6_expand3x3_w.mat' ],  'weights');
    %
    bias = net.params('fire6/expand3x3',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(16));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire6_expand3x3_b.mat' ],  'bias');
    %
    %fire7
    weights = net.params('fire7/squeeze1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(17));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire7_squeeze1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire7/squeeze1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(17));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire7_squeeze1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire7/expand1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(18));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire7_expand1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire7/expand1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(18));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire7_expand1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire7/expand3x3',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(19));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire7_expand3x3_w.mat' ],  'weights');
    %
    bias = net.params('fire7/expand3x3',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(19));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire7_expand3x3_b.mat' ],  'bias');
    %
    %fire8
    weights = net.params('fire8/squeeze1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(20));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire8_squeeze1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire8/squeeze1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(20));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire8_squeeze1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire8/expand1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(21));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire8_expand1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire8/expand1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(21));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire8_expand1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire8/expand3x3',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(22));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire8_expand3x3_w.mat' ],  'weights');
    %
    bias = net.params('fire8/expand3x3',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(22));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire8_expand3x3_b.mat' ],  'bias');
    %
    %fire9
    weights = net.params('fire9/squeeze1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(23));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire9_squeeze1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire9/squeeze1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(23));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire9_squeeze1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire9/expand1x1',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(24));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire9_expand1x1_w.mat' ],  'weights');
    %
    bias = net.params('fire9/expand1x1',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(24));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire9_expand1x1_b.mat' ],  'bias');
    %
    weights = net.params('fire9/expand3x3',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(25));
        weights = storedInteger(weights);
    end
    save( [ param_path 'fire9_expand3x3_w.mat' ],  'weights');
    %
    bias = net.params('fire9/expand3x3',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(25));
        bias = storedInteger(bias);
    end
    save( [ param_path 'fire9_expand3x3_b.mat' ],  'bias');
    %
    %conv10
    weights = net.params('conv10',1).get_data();
    weights = permute(weights, [ 3 1 2 4 ]);
    if (is_fixed==1)
        weights = fi(weights, true, w, f(26));
        weights = storedInteger(weights);
    end
    save( [ param_path 'conv10_w.mat' ],  'weights');
    %
    bias = net.params('conv10',2).get_data();
    if (is_fixed)
        bias = fi(bias, true, w, f(26));
        bias = storedInteger(bias);
    end
    save( [ param_path 'conv10_b.mat' ],  'bias');
end


% Do forward network pass of the caffe-ristretto (quantized) net to get scores
% Initialize the network using the squeezenet_v1.1.caffemodel
net_model  = [model_path q_model_name];
net_params = [model_path 'squeezenet_v1.1.caffemodel'];
phase      = 'test'; % run with phase test (so that dropout isn't applied)

% Initialize the network
net = caffe.Net(net_model, net_params, phase);

%net.blobs('data').reshape([227 227 3 1]); % reshape blob 'data'
%net.reshape();
net.forward({img});


% Get the intermediate results and save them to matlab and binary files
% for checking against Matlab and C/C++ implementations
if ( extract_iresults==1 )

    data = net.blobs('conv1').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '2_conv1.mat' ], 'data');
    var_to_bin(data, [bin_path '2_conv1.bin'], 'w', 'single');
    %
    data = net.blobs('pool1').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '3_pool1.mat' ], 'data');
    var_to_bin(data, [bin_path '3_pool1.bin'], 'w', 'single');
    %
    data = net.blobs('fire2/concat').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '4_fire2.mat' ], 'data');
    var_to_bin(data, [bin_path '4_fire2.bin'], 'w', 'single');
    %
    data = net.blobs('fire3/concat').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '5_fire3.mat' ], 'data');
    var_to_bin(data, [bin_path '5_fire3.bin'], 'w', 'single');
    %
    data = net.blobs('pool3').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '6_pool3.mat' ], 'data');
    var_to_bin(data, [bin_path '6_pool3.bin'], 'w', 'single');
    %
    data = net.blobs('fire4/concat').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '7_fire4.mat' ], 'data');
    var_to_bin(data, [bin_path '7_fire4.bin'], 'w', 'single');
    %
    data = net.blobs('fire5/concat').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '8_fire5.mat' ], 'data');
    var_to_bin(data, [bin_path '8_fire5.bin'], 'w', 'single');
    %
    data = net.blobs('pool5').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '9_pool5.mat' ], 'data');
    var_to_bin(data, [bin_path '9_pool5.bin'], 'w', 'single');
    %
    data = net.blobs('fire6/concat').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '10_fire6.mat' ], 'data');
    var_to_bin(data, [bin_path '10_fire6.bin'], 'w', 'single');
    %
    data = net.blobs('fire7/concat').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '11_fire7.mat' ], 'data');
    var_to_bin(data, [bin_path '11_fire7.bin'], 'w', 'single');
    %
    data = net.blobs('fire8/concat').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '12_fire8.mat' ], 'data');
    var_to_bin(data, [bin_path '12_fire8.bin'], 'w', 'single');
    %
    data = net.blobs('fire9/concat').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '13_fire9.mat' ], 'data');
    var_to_bin(data, [bin_path '13_fire9.bin'], 'w', 'single');
    %
    data = net.blobs('conv10').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '14_conv10.mat' ], 'data');
    var_to_bin(data, [bin_path '14_conv10.bin'], 'w', 'single');
    %
    data = net.blobs('pool10').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '15_pool10.mat' ], 'data');
    var_to_bin(data, [bin_path '15_pool10.bin'], 'w', 'single');
    %
    data = net.blobs('prob').get_data();
    data = permute(data,[3 1 2]);
    save( [ ires_path '16_prob.mat' ], 'data');
    var_to_bin(data, [bin_path '16_prob.bin'], 'w', 'single');
end


end