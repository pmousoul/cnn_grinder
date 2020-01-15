% Reshape Layer.
% bottom is a 3d matrix: N x Win x Hin.
% top is a 3d matrix: M x Wout x Hout.
% Kernel size K and stride S are integers.

function [ top ] = layer_reshape( bottom, K, S, M, pad, type )
    [N,Win,Hin]=size(bottom);
    
    % Add padding
    bottomPadded=zeros(N, Win+2*pad,Hin+2*pad, type);
    bottomPadded(:,pad+1:end-pad,pad+1:end-pad)=bottom;
    
    [N,Win_p,Hin_p]=size(bottomPadded);
    
    % Set output feature map dimensions
    Wout=floor( (Win_p-K)/S+1 );
    Hout=floor( (Hin_p-K)/S+1 );
    top=zeros(M,Wout,Hout, type);
    
    % Reshape
    for w=1:Wout
        for h=1:Hout
            wStart=(w-1)*S+1;
            wEnd=wStart+K-1;
            hStart=(h-1)*S+1;
            hEnd=hStart+K-1;
            temp = bottomPadded(1:N,wStart:wEnd,hStart:hEnd);
            top(1:(K*K*N),w,h)= temp(:);
            top((K*K*N+1):M,w,h) = 0;
        end
    end
end
