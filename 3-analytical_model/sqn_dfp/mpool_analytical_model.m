% This script evaluates the analytical model of the maxpool
% accelerator used for the acceleration of the SqueezeNet CNN.
% Pipeline depths are taken from the Xilinx HLS
% performance estimation.

function [ HW_MPOOL_CC ] = mpool_analytical_model(H_IN, W_IN, CH, K, STRIDE, PAD)

% Paramaters of the maxpool operation

% These parameters change in each layer iteration
% and are controlled by software
H_OUT = floor( ( H_IN - K + PAD )/STRIDE + 1 );
W_OUT = floor( ( W_IN - K + PAD )/STRIDE + 1 );

% These parameters are hardware parameters
FM_PACK = 8;

PIPE_CO_FILL  = 2;

PIPE_CWC_FILL = 3;

PIPE_RC_FILL  = 3;

PIPE_BP_FILL  = 2;


MPOOL_OVER = 3;

BP_OVER    = 2;

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
L_CO_CC  = ( CH / FM_PACK ) + PIPE_CO_FILL + CO_OVER;

% Calc column-wise max cycle count
L_CWC_CC = ( CH / FM_PACK ) + PIPE_CWC_FILL + CWC_OVER;
L_CWW_CC = K * L_CWC_CC + CWW_OVER;
L_CW_CC  = W_OUT * ( L_CWW_CC + L_CO_CC ) + CW_OVER;

% Calc row-wise max cycle count
L_RC_CC  = ( ( W_IN * CH )/ FM_PACK ) + PIPE_RC_FILL + RC_OVER;
L_RH_CC  = K * L_RC_CC + RH_OVER;

% Calc bypass logic cycle count
L_BP_CC  = ( ( H_OUT * W_OUT * CH )/ FM_PACK ) + PIPE_BP_FILL + BP_OVER;

% Calc maxpool operation total cycle count
L_MP_CC  = H_OUT * ( L_RH_CC + L_CW_CC );

if (K==1)
    HW_MPOOL_CC = L_BP_CC + MPOOL_OVER;
else
    HW_MPOOL_CC = L_MP_CC + MPOOL_OVER;
end

end














