% Write variable to binary file

function svar_to_bfile( var, filename, open_file_type, write_value_type)

    fileID = fopen(filename, open_file_type);
    fwrite(fileID, var, write_value_type);
    fclose(fileID);

end