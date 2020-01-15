% This script evaluates the analytical model of the merged
% convolutional-maxpool accelerator used for the acceleration
% of the SqueezeNet and the ZynqNet CNNs. Pipeline depths are taken
% from the Xilinx HLS performance estimation.

function [ HW_CONV_MPOOL_CC ] = conv_mpool_analytical_model(H_IN, W_IN, CHI, CHO, K, STRIDE, PAD, pK, pSTRIDE, pPAD)


HW_CONV_CC = conv_analytical_model(H_IN, W_IN, CHI, CHO, K, STRIDE, PAD);

% Paramaters of the convolution operation
% These parameters change in each layer iteration
% and are controlled by software
H_OUT = floor( ( H_IN - K + 2*PAD )/STRIDE + 1 );
W_OUT = floor( ( W_IN - K + 2*PAD )/STRIDE + 1 );

% conv-related latency in cycles
% The conv accelerator produces an output channel every PC_CC cycles
% These parameters are hardware parameters
PAR_FACT=16;
CHI_NUM =16;
FM_PACK = 8;
WB_PACK = 8;

WIxCHI_MAX = 8448;

% Iteration latency reported by HLS
PIPE_CCO_DSP_LUT_FILL = 12;
PIPE_WP_LOOP_FILL     = 1;

PIPE_ILW_FILL         = 5;

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

ILW_OVER  = 1;

FL_OVER   = 1;
FLLR_OVER = 1;

CL_OVER   = 2;
IL_OVER   = 2;

IP_OVER   = 1;

% pixel_calc() cycle count
% the write_pix() function and its loop are also pipelined because of the
% dataflow pragma
CCO_CC = ( CHO * K * K * CHI ) / ( PAR_FACT * CHI_NUM ) + PIPE_CCO_DSP_LUT_FILL + CCO_DSP_LUT_OVER;
WP_CC = PAR_FACT*( ( CHO / PAR_FACT ) + PIPE_WP_LOOP_FILL) + WP_OVER;
PC_CC = CCO_CC + WP_CC - ( ( CHO / PAR_FACT ) + PIPE_WP_LOOP_FILL);

% init_linebuf_win() cycle count
ILW_CC = (K * K * CHI) / ( CHI_NUM ) + PIPE_ILW_FILL + ILW_OVER;

% shift_linebuf() cycle count and H_OUT loop cycle count
CL_CC   = ( WIxCHI_MAX / CHI_NUM ) + PIPE_CL_FILL;
FL_CC   = ( ( W_IN * CHI ) / FM_PACK ) + PIPE_FL_FILL + FL_OVER;
FLLR_CC = ( ( K * CHI ) / FM_PACK ) + PIPE_FLLR_FILL + FLLR_OVER;
if ( STRIDE==2 )
    SL_CC      = FL_CC + FLLR_CC;
else
    SL_CC      = FLLR_CC;
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



% Paramaters of the maxpool operation

% These parameters change in each layer iteration
% and are controlled by software
PH_OUT = floor( ( H_OUT - pK + pPAD )/pSTRIDE + 1 );
PW_OUT = floor( ( W_OUT - pK + pPAD )/pSTRIDE + 1 );

% These parameters are hardware parameters
PIPE_CO_FILL  = 2;

PIPE_CWC_FILL = 3;

PIPE_RC_FILL  = 3;


MPOOL_OVER = 3;

RH_OVER    = 1;

RC_OVER    = 1;

CW_OVER    = 1;

CWW_OVER   = 1;

CWC_OVER   = 2;

CO_OVER    = 1;


% Calculation of the cycle cost of the accelerator's functions/loops
% In general the calculation is:
% latency = trip_count*II + IL (for pipelined loops)
% latency = trip_count*cycles_of_the_loop_body (for non-pipelined loops)

% Write-back calculated pixel cycle count
L_CO_CC  = ( CHO / FM_PACK ) + PIPE_CO_FILL + CO_OVER;


% Calc column-wise max cycle count
L_CWC_CC = ( CHO / FM_PACK ) + PIPE_CWC_FILL + CWC_OVER;
L_CWW_CC = pK * L_CWC_CC + CWW_OVER;

L_CW_CC  = PW_OUT * ( L_CWW_CC + L_CO_CC ) + CW_OVER;


% Calc row-wise max cycle count

% Cycle count required by the conv module to produce a row of output pixels
L_RCM_CC  = W_OUT * PC_CC;

% Cycle count required by the mpool module to read a row of pixels
% L_RC_CC  = ( ( W_OUT * CHO ) / FM_PACK ) + PIPE_RC_FILL + RC_OVER;
% We assume that the mpool module consumes a pixel by the time the conv module does to produce a new one
% So, the mpool module will wait only for the last pixel
L_RC_CC  = ( ( 1 * CHO ) / FM_PACK ) + PIPE_RC_FILL + RC_OVER;

% The mpool module reads pK rows (produced by the conv module) only the first time it runs
% All the other times it reads only ( pK - 1 ) 
L_RHM_CC  = ( ( pK - 1 ) * ( L_RC_CC + L_RCM_CC ) ) + RH_OVER;


% Calc maxpool operation total cycle count
L_MPM_CC  = PH_OUT * ( L_RHM_CC + L_CW_CC );

if (pK==1)

    % We assume that mpool module will wait to write-back only for the last pixel
    HW_CONV_MPOOL_CC = HW_CONV_CC + L_CO_CC + MPOOL_OVER;

else

    HW_CONV_MPOOL_CC = PT_CC + IC_CC + ( SL_CC +  ILW_CC ) + ( W_OUT * PC_CC ) + L_RC_CC +... % Convolution overhead
                        L_MPM_CC + MPOOL_OVER;

end


end