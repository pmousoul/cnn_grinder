% Convolution Layer - caffe version.
% bottom is a 3d matrix: Win x Hin x N.
% top is a 3d matrix: Wout x Hout x M.
% weight is a 4d matrix: K x K x N x M.
% bias is a 1d matrix: M x 1.
% Kernel size K and stride S are integers.
% Padding 'pad' specifies the number of pixels to (implicitly) add to each
% side of the input.

function [ top ] = layer_conv_caffe( bottom, weight, bias, K, S, pad )
    [Win,Hin,N]=size(bottom);
    [~,~,~,M]=size(weight);
    
    % Add padding
    bottomPadded=zeros(Win+2*pad,Hin+2*pad,N);
    bottomPadded(pad+1:end-pad,pad+1:end-pad,:)=bottom;
    
    % Set ofm dimensions
    Wout=floor( (Win+2*pad-K)/S+1 );
    Hout=floor( (Hin+2*pad-K)/S+1 );
    top=zeros(Wout,Hout,M);
    
    % Perform convolution
    for w=1:Wout
        for h=1:Hout
            for m=1:M
                wStart=(w-1)*S+1;
                wEnd=wStart+K-1;
                hStart=(h-1)*S+1;
                hEnd=hStart+K-1;
                top(w,h,m)=top(w,h,m)+sum(sum(sum( ...
                    bottomPadded(wStart:wEnd,hStart:hEnd,1:N) .* ...
                    weight(:,:,1:N,m) )));
            end
        end
    end
    
    % Add bias
    for m=1:M
        top(:,:,m)=top(:,:,m)+bias(m, 1);
    end
end
