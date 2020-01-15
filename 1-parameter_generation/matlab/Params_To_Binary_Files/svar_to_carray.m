% Convert matlab array variable to C array
% Arrays are assumed to be in the following shapes:
%             1   2   3  4
% weights -> [N  KW  KH  M]
% fmap    -> [N   W   H]
% bias    -> [M]

function [var_out] = svar_to_carray( var_in )

    % number of dimensions of input array
    nd = ndims(var_in);
    
    switch nd
        
        case 2 % bias data
            
            % save data in row-major format
            var_in = reshape(var_in,[],1);
            
        case 3 % image data
            
            % save data in row-major format
            var_in = permute(var_in,[3 1 2]);
            var_in = reshape(var_in,size(var_in,1),[],1);
            var_in = var_in';
            var_in = reshape(var_in,[],1);
            
        case 4 % weights data
            
            % save data in row-major format
            var_in = permute(var_in,[4 3 1 2]);
            var_in = reshape(var_in,size(var_in,1),size(var_in,2),[]);
            var_in = permute(var_in,[1 3 2]);
            var_in = reshape(var_in,size(var_in,1),[],1);
            var_in = var_in';
            var_in = reshape(var_in,[],1);
            
            
        otherwise
            
            fprintf('\nERROR: Conversion is not supported.\n');
            
    end
     
    var_out = var_in;

end