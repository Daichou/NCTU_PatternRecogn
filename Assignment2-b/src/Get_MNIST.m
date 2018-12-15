function images = Get_MNIST(filename)
    fp = fopen(filename, 'rb');

    magic = fread(fp, 1, 'int32', 0, 'ieee-be')
    numImages = fread(fp, 1, 'int32', 0, 'ieee-be')
    numRows = fread(fp, 1, 'int32', 0, 'ieee-be')
    numCols = fread(fp, 1, 'int32', 0, 'ieee-be')

    images = fread(fp, inf, 'unsigned char');
    images = reshape(images, numCols, numRows, numImages);
    % rotate image
    images = permute(images,[2 1 3]);

    fclose(fp);

    images = reshape(images, numCols*numRows, numImages);
    images = uint8(images)/255 ;
end
