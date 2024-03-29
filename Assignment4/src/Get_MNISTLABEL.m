function labels = Get_MNISTLABEL(filename)
    fp = fopen(filename, 'rb');
    assert(fp ~= -1, ['Could not open ', filename, '']);
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', filename, '']);
    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be')
    labels = fread(fp, inf, 'uint8');
    assert(size(labels,1) == numLabels, 'Mismatch in label count');
    fclose(fp);
end

