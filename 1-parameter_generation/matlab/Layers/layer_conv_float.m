% Convolution Layer - floating-point version.
% bottom is a 3d matrix: N x Win x Hin.
% top is a 3d matrix: M x Wout x Hout.
% weight is a 4d matrix: N x K x K x M.
% bias is a 1d matrix: M x 1.
% Kernel size K and stride S are integers.
% Padding 'pad' specifies the number of pixels to (implicitly) add to each
% side of the input.

function [ top ] = layer_conv_float( bottom, weight, bias, K, S, pad )
    [N,Win,Hin]=size(bottom);
    [~,~,~,M]=size(weight);
    
    % Add padding
    bottomPadded=zeros(N,Win+2*pad,Hin+2*pad,1, 'single');
    bottomPadded(:,pad+1:end-pad,pad+1:end-pad,1)=bottom;
    
    % Set ofm dimensions
    Wout=floor(  (Win+2*pad-K)/S+1 );
    Hout=floor( (Hin+2*pad-K)/S+1 );
    top=zeros(M,Wout,Hout, 'single');
    
    % Perform convolution
    for h=1:Hout
        for w=1:Wout
            for m=1:M
                wStart=(w-1)*S+1;
                wEnd=wStart+K-1;
                hStart=(h-1)*S+1;
                hEnd=hStart+K-1;
                top(m,w,h)=top(m,w,h)+sum(sum(sum( ...
                    bottomPadded(1:N,wStart:wEnd,hStart:hEnd,1) .* ...
                    weight(1:N,:,:,m) )));
            end
        end
    end
    
    % Add bias
    for m=1:M
        top(m,:,:)=top(m,:,:)+bias(m, 1);
    end
end
