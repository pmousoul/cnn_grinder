% Maxpool over a window of K*K.
% bottom is a 3d matrix: Win x Hin x N.
% top is a 3d matrix: Wout x Hout x N.
% The kernel size K and stride S are integers.
% Pool the input (bottom) with windows of size K and with the specified stride.

function [ top ] = layer_maxpool_caffe( bottom, K, S, pad )
    [Win,Hin,N]=size(bottom);
    
    %Add padding.
    bottomPadded=zeros(Win+pad,Hin+pad,N, 'single');
    bottomPadded(1:end-pad,1:end-pad,:)=bottom;
    [Win,Hin,N]=size(bottomPadded);

    Wout = (Win-K)/S+1;
    Hout = (Hin-K)/S+1;
    top=zeros(Wout,Hout,N, 'single');

    for n=1:N
        for h=1:Hout
            for w=1:Wout
                hstart = (h-1)*S+1;
                wstart = (w-1)*S+1;
                hend=hstart+K-1;
                wend=wstart+K-1;
                top(w,h,n)=max(max(bottomPadded(wstart:wend,hstart:hend,n)));
            end
        end
    end
end
