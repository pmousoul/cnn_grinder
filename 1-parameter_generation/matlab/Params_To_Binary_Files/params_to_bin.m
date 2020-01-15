function [] = params_to_bin(weights, bias, param_file, write_mode, value_type_p)

var1 = svar_to_carray(weights);
var2 = svar_to_carray(bias);
var = [ var1; var2 ];
svar_to_bfile(var, param_file, write_mode, value_type_p);

end