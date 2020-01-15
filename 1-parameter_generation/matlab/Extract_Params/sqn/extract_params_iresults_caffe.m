% Extract layer parameters and intermediate network results from
% SqueezeNet v1.1 CNN caffe version using matcaffe.

function extract_params_iresults_caffe(extract_params, extract_iresults, img_file, model_path, param_path, ires_path)


% Set caffe mode
%caffe.set_mode_cpu();


% Initialize the network using squeezenet_v1.1.caffemodel
net_model = [model_path 'deploy.prototxt'];
net_weights = [model_path 'squeezenet_v1.1.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)


% Initialize the network
net = caffe.Net(net_model, net_weights, phase);


% Get the network parameters and save them to param_path
if ( extract_params==1 )
    % conv1
    weights = net.params('conv1',1).get_data();
    save( [ param_path 'conv1_w.mat' ],  'weights');
    bias = net.params('conv1',2).get_data();
    save( [ param_path 'conv1_b.mat' ],  'bias');

    %fire2
    weights = net.params('fire2/squeeze1x1',1).get_data();
    save( [ param_path 'fire2_squeeze1x1_w.mat' ],  'weights');
    bias = net.params('fire2/squeeze1x1',2).get_data();
    save( [ param_path 'fire2_squeeze1x1_b.mat' ],  'bias'); 
    weights = net.params('fire2/expand1x1',1).get_data();
    save( [ param_path 'fire2_expand1x1_w.mat' ],  'weights');
    bias = net.params('fire2/expand1x1',2).get_data();
    save( [ param_path 'fire2_expand1x1_b.mat' ],  'bias');
    weights = net.params('fire2/expand3x3',1).get_data();
    save( [ param_path 'fire2_expand3x3_w.mat' ],  'weights');
    bias = net.params('fire2/expand3x3',2).get_data();
    save( [ param_path 'fire2_expand3x3_b.mat' ],  'bias');

    %fire3
    weights = net.params('fire3/squeeze1x1',1).get_data();
    save( [ param_path 'fire3_squeeze1x1_w.mat' ],  'weights');
    bias = net.params('fire3/squeeze1x1',2).get_data();
    save( [ param_path 'fire3_squeeze1x1_b.mat' ],  'bias'); 
    weights = net.params('fire3/expand1x1',1).get_data();
    save( [ param_path 'fire3_expand1x1_w.mat' ],  'weights');
    bias = net.params('fire3/expand1x1',2).get_data();
    save( [ param_path 'fire3_expand1x1_b.mat' ],  'bias');
    weights = net.params('fire3/expand3x3',1).get_data();
    save( [ param_path 'fire3_expand3x3_w.mat' ],  'weights');
    bias = net.params('fire3/expand3x3',2).get_data();
    save( [ param_path 'fire3_expand3x3_b.mat' ],  'bias');

    %fire4
    weights = net.params('fire4/squeeze1x1',1).get_data();
    save( [ param_path 'fire4_squeeze1x1_w.mat' ],  'weights');
    bias = net.params('fire4/squeeze1x1',2).get_data();
    save( [ param_path 'fire4_squeeze1x1_b.mat' ],  'bias'); 
    weights = net.params('fire4/expand1x1',1).get_data();
    save( [ param_path 'fire4_expand1x1_w.mat' ],  'weights');
    bias = net.params('fire4/expand1x1',2).get_data();
    save( [ param_path 'fire4_expand1x1_b.mat' ],  'bias');
    weights = net.params('fire4/expand3x3',1).get_data();
    save( [ param_path 'fire4_expand3x3_w.mat' ],  'weights');
    bias = net.params('fire4/expand3x3',2).get_data();
    save( [ param_path 'fire4_expand3x3_b.mat' ],  'bias');

    %fire5
    weights = net.params('fire5/squeeze1x1',1).get_data();
    save( [ param_path 'fire5_squeeze1x1_w.mat' ],  'weights');
    bias = net.params('fire5/squeeze1x1',2).get_data();
    save( [ param_path 'fire5_squeeze1x1_b.mat' ],  'bias'); 
    weights = net.params('fire5/expand1x1',1).get_data();
    save( [ param_path 'fire5_expand1x1_w.mat' ],  'weights');
    bias = net.params('fire5/expand1x1',2).get_data();
    save( [ param_path 'fire5_expand1x1_b.mat' ],  'bias');
    weights = net.params('fire5/expand3x3',1).get_data();
    save( [ param_path 'fire5_expand3x3_w.mat' ],  'weights');
    bias = net.params('fire5/expand3x3',2).get_data();
    save( [ param_path 'fire5_expand3x3_b.mat' ],  'bias');

    %fire6
    weights = net.params('fire6/squeeze1x1',1).get_data();
    save( [ param_path 'fire6_squeeze1x1_w.mat' ],  'weights');
    bias = net.params('fire6/squeeze1x1',2).get_data();
    save( [ param_path 'fire6_squeeze1x1_b.mat' ],  'bias'); 
    weights = net.params('fire6/expand1x1',1).get_data();
    save( [ param_path 'fire6_expand1x1_w.mat' ],  'weights');
    bias = net.params('fire6/expand1x1',2).get_data();
    save( [ param_path 'fire6_expand1x1_b.mat' ],  'bias');
    weights = net.params('fire6/expand3x3',1).get_data();
    save( [ param_path 'fire6_expand3x3_w.mat' ],  'weights');
    bias = net.params('fire6/expand3x3',2).get_data();
    save( [ param_path 'fire6_expand3x3_b.mat' ],  'bias');

    %fire7
    weights = net.params('fire7/squeeze1x1',1).get_data();
    save( [ param_path 'fire7_squeeze1x1_w.mat' ],  'weights');
    bias = net.params('fire7/squeeze1x1',2).get_data();
    save( [ param_path 'fire7_squeeze1x1_b.mat' ],  'bias'); 
    weights = net.params('fire7/expand1x1',1).get_data();
    save( [ param_path 'fire7_expand1x1_w.mat' ],  'weights');
    bias = net.params('fire7/expand1x1',2).get_data();
    save( [ param_path 'fire7_expand1x1_b.mat' ],  'bias');
    weights = net.params('fire7/expand3x3',1).get_data();
    save( [ param_path 'fire7_expand3x3_w.mat' ],  'weights');
    bias = net.params('fire7/expand3x3',2).get_data();
    save( [ param_path 'fire7_expand3x3_b.mat' ],  'bias');

    %fire8
    weights = net.params('fire8/squeeze1x1',1).get_data();
    save( [ param_path 'fire8_squeeze1x1_w.mat' ],  'weights');
    bias = net.params('fire8/squeeze1x1',2).get_data();
    save( [ param_path 'fire8_squeeze1x1_b.mat' ],  'bias'); 
    weights = net.params('fire8/expand1x1',1).get_data();
    save( [ param_path 'fire8_expand1x1_w.mat' ],  'weights');
    bias = net.params('fire8/expand1x1',2).get_data();
    save( [ param_path 'fire8_expand1x1_b.mat' ],  'bias');
    weights = net.params('fire8/expand3x3',1).get_data();
    save( [ param_path 'fire8_expand3x3_w.mat' ],  'weights');
    bias = net.params('fire8/expand3x3',2).get_data();
    save( [ param_path 'fire8_expand3x3_b.mat' ],  'bias');

    %fire9
    weights = net.params('fire9/squeeze1x1',1).get_data();
    save( [ param_path 'fire9_squeeze1x1_w.mat' ],  'weights');
    bias = net.params('fire9/squeeze1x1',2).get_data();
    save( [ param_path 'fire9_squeeze1x1_b.mat' ],  'bias'); 
    weights = net.params('fire9/expand1x1',1).get_data();
    save( [ param_path 'fire9_expand1x1_w.mat' ],  'weights');
    bias = net.params('fire9/expand1x1',2).get_data();
    save( [ param_path 'fire9_expand1x1_b.mat' ],  'bias');
    weights = net.params('fire9/expand3x3',1).get_data();
    save( [ param_path 'fire9_expand3x3_w.mat' ],  'weights');
    bias = net.params('fire9/expand3x3',2).get_data();
    save( [ param_path 'fire9_expand3x3_b.mat' ],  'bias');

    %conv10
    weights = net.params('conv10',1).get_data();
    save( [ param_path 'conv10_w.mat' ],  'weights');
    bias = net.params('conv10',2).get_data();
    save( [ param_path 'conv10_b.mat' ],  'bias');
end


% Do forward network pass to get scores
net.blobs('data').reshape([227 227 3 1]); % reshape blob 'data'
net.reshape();
net.forward(img_file);


% Get the intermediate results and save them to ires_path
if ( extract_iresults==1 )
    data = net.blobs('conv1').get_data();
    save( [ ires_path '2_conv1.mat' ], 'data');

    data = net.blobs('pool1').get_data();
    save( [ ires_path '3_pool1.mat' ], 'data');

    data = net.blobs('fire2/concat').get_data();
    save( [ ires_path '4_fire2.mat' ], 'data');

    data = net.blobs('fire3/concat').get_data();
    save( [ ires_path '5_fire3.mat' ], 'data');

    data = net.blobs('pool3').get_data();
    save( [ ires_path '6_pool3.mat' ], 'data');

    data = net.blobs('fire4/concat').get_data();
    save( [ ires_path '7_fire4.mat' ], 'data');

    data = net.blobs('fire5/concat').get_data();
    save( [ ires_path '8_fire5.mat' ], 'data');

    data = net.blobs('pool5').get_data();
    save( [ ires_path '9_pool5.mat' ], 'data');

    data = net.blobs('fire6/concat').get_data();
    save( [ ires_path '10_fire6.mat' ], 'data');

    data = net.blobs('fire7/concat').get_data();
    save( [ ires_path '11_fire7.mat' ], 'data');

    data = net.blobs('fire8/concat').get_data();
    save( [ ires_path '12_fire8.mat' ], 'data');

    data = net.blobs('fire9/concat').get_data();
    save( [ ires_path '13_fire9.mat' ], 'data');

    data = net.blobs('conv10').get_data();
    save( [ ires_path '14_conv10.mat' ], 'data');

    data = net.blobs('pool10').get_data();
    save( [ ires_path '15_pool10.mat' ], 'data');

    data = net.blobs('prob').get_data();
    save( [ ires_path '16_prob.mat' ], 'data');
end


end