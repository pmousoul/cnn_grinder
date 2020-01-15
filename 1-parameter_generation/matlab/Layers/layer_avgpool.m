% Avgpool over a window of K*K.
% bottom is a 3d matrix: Win x Hin x N.
% top is a 3d matrix: Wout x Hout x N.
% The kernel size K and stride S are integers.
% Pool the input (bottom) with windows of size K and with the specified stride.
% No padding needed.

function [ top ] = layer_avgpool( bottom, K, S )
    [N,Win,Hin]=size(bottom);

    Wout = (Win-K)/S+1;
    Hout = (Hin-K)/S+1;
    top=zeros(N,Wout,Hout);

    for h=1:Hout
        for w=1:Wout
            for n=1:N
                hstart = (h-1)*S+1;
                wstart = (w-1)*S+1;
                hend=hstart+K-1;
                wend=wstart+K-1;
                top(n,w,h)=mean(mean(bottom(n,wstart:wend,hstart:hend)));
            end
        end
    end
end
