% Maxpool over a window of K*K.
% bottom is a 3d matrix: N x Win x Hin.
% top is a 3d matrix: N x Wout x Hout.
% The kernel size K and stride S are integers.
% Pool the input (bottom) with windows of size K and with the specified stride.

function [ top ] = layer_maxpool( bottom, K, S, pad, type )
    [N,Win,Hin]=size(bottom);
    
    %Add padding.
    bottomPadded=zeros(N,Win+pad,Hin+pad, type);
    bottomPadded(:,1:end-pad,1:end-pad)=bottom;
    [N,Win,Hin]=size(bottomPadded);

    Wout = (Win-K)/S+1;
    Hout = (Hin-K)/S+1;
    top=zeros(N,Wout,Hout, type);

    for n=1:N
        for w=1:Wout
            for h=1:Hout
                hstart = (h-1)*S+1;
                wstart = (w-1)*S+1;
                hend=hstart+K-1;
                wend=wstart+K-1;
                top(n,w,h)=max(max(bottomPadded(n,wstart:wend,hstart:hend)));
            end
        end
    end
end
