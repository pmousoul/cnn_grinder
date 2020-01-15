% This script evaluates the analytical model of the convolutional
% accelerator used for the acceleration of the SqueezeNet and the
% ZynqNet CNNs. Pipeline depths are taken from the Xilinx HLS
% performance estimation.

function [ HW_CONV_CC ] = conv_analytical_model(H_IN, W_IN, CHI, CHO, K, STRIDE, PAD)

% Paramaters of the convolution operation
% These parameters change in each layer iteration
% and are controlled by software
H_OUT = floor( ( H_IN - K + 2*PAD )/STRIDE + 1 );
W_OUT = floor( ( W_IN - K + 2*PAD )/STRIDE + 1 );

% These parameters are hardware parameters
PAR_FACT=16;
CHI_NUM =16;
FM_PACK = 8;
WB_PACK = 8;

WIxCHI_MAX = 8448;


% Iteration latency reported by HLS
PIPE_CCO_DSP_LUT_FILL = 12;
PIPE_WP_LOOP_FILL     = 1;

PIPE_FLP_LOOP_FILL    = 2;
PIPE_ILW_FILL         = 5;

PIPE_WB_LOOP_FILL     = 3;

PIPE_CL_FILL          = 1;
PIPE_FL_FILL          = 2;
PIPE_FLLR_FILL        = 2;

PIPE_IP_LL_W_FILL     = 2;
PIPE_IP_LL_BT_FILL    = 2;
PIPE_IP_LL_B_FILL     = 2;


% Overhead reported by HLS
% we include the READ_BIAS loop in the CCO_DSP_LUT_OVER
CCO_DSP_LUT_OVER  = 2;
WP_OVER   = 1;

FLP_OVER  = 1;
ILW_OVER  = 1;
ULW_OVER  = 2;

WB_OVER   = 1;

FL_OVER   = 1;
FLLR_OVER = 1;

CL_OVER   = 2;
IL_OVER   = 2;

IP_OVER   = 1;


% Calculation of the cycle cost of the accelerator's functions/loops
% In general the calculation is:
% latency = trip_count*II + IL (for pipelined loops)
% latency = trip_count*cycles_to_read_data (for non-pipelined loops)

% pixel_calc() cycle count
% the write_pix() function and its loop are also pipelined because of the
% dataflow pragma
CCO_CC = ( CHO * K * K * CHI ) / ( PAR_FACT * CHI_NUM ) + PIPE_CCO_DSP_LUT_FILL + CCO_DSP_LUT_OVER;
WP_CC = PAR_FACT*( ( CHO / PAR_FACT ) + PIPE_WP_LOOP_FILL) + WP_OVER;
PC_CC = CCO_CC + WP_CC - ( ( CHO / PAR_FACT ) + PIPE_WP_LOOP_FILL);

% update_linebuf_win() cycle count
FLP_CC = ( ( STRIDE * CHI ) / FM_PACK ) + PIPE_FLP_LOOP_FILL + FLP_OVER;
ILW_CC = (K * K * CHI) / ( CHI_NUM ) + PIPE_ILW_FILL + ILW_OVER;
ULW_CC = FLP_CC + ILW_CC + ULW_OVER;

% write_back() cycle count
WB_CC  = ( CHO / FM_PACK ) + PIPE_WB_LOOP_FILL + WB_OVER;

% W_OUT loop cycle count
% the PC_CC factor always dominates
L_W_OUT_CC = W_OUT * ( max([ PC_CC, ULW_CC, WB_CC ]) );

% shift_linebuf() cycle count and H_OUT loop cycle count
CL_CC   = ( WIxCHI_MAX / CHI_NUM ) + PIPE_CL_FILL;
FL_CC   = ( ( W_IN * CHI ) / FM_PACK ) + PIPE_FL_FILL + FL_OVER;
FLLR_CC = ( ( K * CHI ) / FM_PACK ) + PIPE_FLLR_FILL + FLLR_OVER;
if ( STRIDE==2 )
    SL_CC      = FL_CC + FLLR_CC;
    L_H_OUT_CC = ( H_OUT * ( SL_CC +  ILW_CC + L_W_OUT_CC ) ) - FL_CC;
else
    SL_CC      = FLLR_CC;
    L_H_OUT_CC = ( H_OUT * ( SL_CC +  ILW_CC + L_W_OUT_CC ) ) - FLLR_CC + ( PAD * CL_CC );
end

% init_caches() cycle count
% init_linebufs()
if( PAD==0 )
    IL_CC = 0 + IL_OVER;
else
    IL_CC = K * ( CL_CC + CL_OVER ) + FL_CC + IL_OVER;
end
LL_W_CC  = PAR_FACT * ( (CHO * K * K * CHI ) / ( PAR_FACT * WB_PACK ) + PIPE_IP_LL_W_FILL );
LL_BT_CC = ( CHO / WB_PACK ) + PIPE_IP_LL_BT_FILL;
LL_B_CC  = PAR_FACT * ( CHO/PAR_FACT + PIPE_IP_LL_B_FILL);
% init_params()
IP_CC    = LL_W_CC + LL_BT_CC + LL_B_CC + IP_OVER;
% init_caches()
IC_CC= max( [ IL_CC, IP_CC ] );

% precalc_terms() cycle count
PT_CC = 3;

HW_CONV_CC = PT_CC + IC_CC + L_H_OUT_CC;

end

