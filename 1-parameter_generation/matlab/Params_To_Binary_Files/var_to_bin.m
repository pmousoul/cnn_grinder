function [] = var_to_bin(data, file, write_mode, value_type_p)

var = svar_to_carray(data);
svar_to_bfile(var, file, write_mode, value_type_p);

end