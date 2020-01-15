function [inputFile, meanFile, cmp, matcaffePath, bin] =read_config(config_file)

fileId = fopen(config_file);
if (fileId == -1)
    error('Cannot find config.txt in the current directory');
end

line = fgets(fileId);
while (ischar(line)) 
    tokens = strsplit(line, '=');
    if (line(1) == '#')
        line = fgets(fileId);
        continue;
    end
    switch(strtrim(tokens{1}))
        case 'input_file'
            inputFile = strtrim(tokens{2});
        case 'mean_file'
            meanFile = strtrim(tokens{2});
        case 'cmp'
            cmp = str2double(strtrim(tokens{2}));
        case 'matcaffe_path'
            matcaffePath = strtrim(tokens{2});
        case 'bin'
            bin = str2double(strtrim(tokens{2}));
    end
    line = fgets(fileId);
end
fclose(fileId);

end