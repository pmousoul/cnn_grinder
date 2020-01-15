% Avgpool over a window of K*K - caffe version.
% bottom is a 3d matrix: Win x Hin x N.
% top is a 3d matrix: Wout x Hout x N.
% The kernel size K and stride S are integers.
% Pool the input (bottom) with windows of size K and with the specified stride.
% No padding needed.

function [ top ] = layer_avgpool_caffe( bottom, K, S )
    [Win,Hin,N]=size(bottom);

    Wout = (Win-K)/S+1;
    Hout = (Hin-K)/S+1;
    top=zeros(Wout,Hout,N);

    for n=1:N
        for h=1:Hout
            for w=1:Wout
                hstart = (h-1)*S+1;
                wstart = (w-1)*S+1;
                hend=hstart+K-1;
                wend=wstart+K-1;
                top(w,h,n)=mean(mean(bottom(wstart:wend,hstart:hend,n)));
%                 for wh=hstart:hend
%                     for ww=wstart:wend
%                         top(w,h,n)=top(w,h,n)+bottom(ww,wh,n);
%                     end
%                 end
%                 top(w,h,n)=top(w,h,n) / (K^2);
            end
        end
    end
end
