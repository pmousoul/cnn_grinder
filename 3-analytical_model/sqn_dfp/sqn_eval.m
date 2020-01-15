% SqueezeNet v1.1 dynammic fixed-point layer latency evaluation

clear all;
clc;


h_in    = [113 56 56 56  56 56 56  28  28  28  28  28  28  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14];
w_in    = [113 56 56 56  56 56 56  28  28  28  28  28  28  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14  14];
ch_in   = [ 32 64 16 16 128 16 16 128  32  32 256  32  32 256  48  48 384  48  48 384  64  64 512  64  64 512 512 512 512];
ch_out  = [ 64 16 64 64  16 64 64  32 128 128  32 128 128  48 192 192  48 192 192  64 256 256  64 256 256 256 256 256 256];
pad     = [  0  0  0  1   0  0  1   0   0   1   0   0   1   0   0   1   0   0   1   0   0   1   0   0   1   0   0   0   0];
kernel  = [  1  1  1  3   1  1  3   1   1   3   1   1   3   1   1   3   1   1   3   1   1   3   1   1   3   1   1   1   1];
stride  = [  1  1  1  1   1  1  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1];
ppad    = [  0  0  0  0   0  1  1   0   0   0   0   1   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0];
pkernel = [  3  1  1  1   1  3  3   1   1   1   1   3   3   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1];
pstride = [  2  1  1  1   1  2  2   1   1   1   1   2   2   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1];


% calculate latency using the analytical model
latency_conv       = zeros([29 1]);
latency_mpool      = zeros([29 1]);
latency_conv_mpool = zeros([29 1]); 

for i=1:29
    
    % latency in ms
    latency_conv(i)       = conv_analytical_model( h_in(i), w_in(i), ch_in(i), ch_out(i), kernel(i), stride(i), pad(i) ) / 10^5;
    latency_mpool(i)      = mpool_analytical_model( h_in(i), w_in(i), ch_out(i), pkernel(i), pstride(i), ppad(i) ) / 10^5;
    latency_conv_mpool(i) = conv_mpool_analytical_model( h_in(i), w_in(i), ch_in(i), ch_out(i), kernel(i), stride(i), pad(i), pkernel(i), pstride(i), ppad(i) ) / 10^5;
    
end




% read files holding the accelerator latency from hardware trace reading
% and from c-rtl co-simulation
fileID = fopen('trace_latencies.txt','r');
formatSpec = '%f';
L_TRACE= fscanf(fileID,formatSpec);
%L_TRACE= L_TRACE /10^5;

fileID = fopen('simulation_latencies.txt','r');
formatSpec = '%f';
L_COSIM= fscanf(fileID,formatSpec);
%L_COSIM= L_COSIM /10^5;


% plot the accelerator latency of each measurement method side-by-side
%figure(7)
subplot(2,1,1);
lat = bar([ L_TRACE L_COSIM latency_conv_mpool ], 1);
title('(A) SqueezeJet-2-dfp accelerator latency comparison')
xlabel('SqueezeNet v1.1 layer number') 
ylabel('latency in ms')
grid on
set(lat(1), 'FaceColor','r')
set(lat(2), 'FaceColor','b')
set(lat(3), 'FaceColor','g')
legend(lat(:), {'L\_TRACE' 'L\_COSIM' 'L\_MODEL'})

% plot the accelerator latency % error of the measurement methods against the trace method
%figure(9)
subplot(2,1,2);
temp1 = abs( L_TRACE-L_COSIM )   ./ L_TRACE .* 100;
temp3 = abs( L_TRACE-latency_conv_mpool )   ./ L_TRACE .* 100;
laterrper = bar([ temp1 temp3 ], 1);
title('(B) SqueezeJet-2-dfp accelerator % latency error VS the Trace method')
xlabel('SqueezeNet v1.1 layer number') 
ylabel('% latency error')
grid on
set(laterrper(1), 'FaceColor','b')
set(laterrper(2), 'FaceColor','g')
legend(laterrper(:), {'L\_COSIM' 'L\_MODEL'})

% print the mean value of
fprintf( 'Conv-Mpool accelerator %% latency error against the Trace method:\n' )
fprintf( 'Mean of the %% L_MODEL error is equal to %f \n', mean(temp3) )
fprintf( 'Mean of the %% L_COSIM error is equal to %f \n', mean(temp1) )
fprintf( 'Max of the %% L_MODEL error is equal to %f \n', max(temp3) )
fprintf( 'Max of the %% L_COSIM error is equal to %f \n', max(temp1) )
fprintf( 'Var of the %% L_MODEL error is equal to %f \n', var(temp3) )
fprintf( 'Var of the %% L_COSIM error is equal to %f \n', var(temp1) )

total_latency=0;
% Calculate total latency in ms
for i=1:29
    % latency in ms
    total_latency = total_latency + latency_conv_mpool(i);
end
fprintf( 'Total latency in ms is equal to %f \n', total_latency );