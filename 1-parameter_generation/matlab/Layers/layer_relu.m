% ReLU Nonlinearity: Rectified Linear Unit.
% Formula: top=max(0,bottom).

function [ top ] = layer_relu( bottom )
    top=max(0,bottom);
end
