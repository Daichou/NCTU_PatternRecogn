% HW1_P20
figure;
% load data
filename = 'train-images-idx3-ubyte'
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

num=randi(length(images),10,15)

for i=0:9
    for j=0:14
        img(numCols*i+1:numCols*(i+1),numRows*j+1:numRows*(j+1))=reshape(images(:,num(i+1,j+1)),numCols,numRows);
    end
end

imshow(img)