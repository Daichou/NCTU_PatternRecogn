clear all;
trInput = Get_MNIST('train-images-idx3-ubyte');
tsInput = Get_MNIST('t10k-images-idx3-ubyte');
trDes = Get_MNISTLABEL('train-labels-idx1-ubyte');
tsDes = Get_MNISTLABEL('t10k-labels-idx1-ubyte');
n_class = 10;

trInput = reshape(trInput,28,28,1,60000);
tsInput = reshape(tsInput,28,28,1,10000);
trInput = trInput(:,:,:,1:1000);
tsInput = tsInput(:,:,:,1:1000);

trDes = trDes(1:1000);
tsDes = tsDes(1:1000);

learning_rate = 0.005;

layer1_filter_num = 6;
layer2_filter_num = 12;
filter_size = 5;
num_class = 10;
pooling2_size = layer2_filter_num*16;
image_size = 28;
conv_1_size = image_size+1-filter_size;
conv_2_size = (image_size+1-filter_size)/2 + 1 -filter_size;
pool_1_size = conv_1_size/2;
pool_2_size = conv_2_size/2;

b1_layer = zeros(layer1_filter_num,1);
b2_layer = zeros(layer2_filter_num,1);
k1_layer = rand(layer1_filter_num,filter_size,filter_size)*sqrt(layer1_filter_num/((1+layer1_filter_num)*filter_size*filter_size));
k2_layer = rand(layer1_filter_num,layer2_filter_num,filter_size,filter_size)*sqrt(layer1_filter_num/((layer2_filter_num+layer1_filter_num)*filter_size*filter_size));
FC_W = rand(num_class,pooling2_size)*sqrt(layer1_filter_num/(pooling2_size+num_class));
FC_b = zeros(num_class,1);
conv_1 = zeros(layer1_filter_num,conv_1_size,conv_1_size);
conv_2 = zeros(layer2_filter_num,layer1_filter_num,conv_2_size,conv_2_size);
sigmoid_1 = zeros(layer1_filter_num,conv_1_size,conv_1_size);
sigmoid_2 = zeros(layer2_filter_num,conv_2_size,conv_2_size);
S1_layer = zeros(layer1_filter_num,pool_1_size,pool_1_size);
S2_layer = zeros(layer2_filter_num,pool_2_size,pool_2_size);
FC_layer = zeros(num_class,1);

%back propagation parameter
delta_W = zeros(num_class,pool_2_size);
delta_B = zeros(num_class,1);
delta_k2 = zeros(layer1_filter_num,layer2_filter_num,filter_size,filter_size);
delta_b2 = zeros(layer2_filter_num,1);
delta_k1 = zeros(layer1_filter_num,filter_size,filter_size);
delta_b1 = zeros(layer1_filter_num,1);
delta_S2 = zeros(layer2_filter_num,pool_2_size,pool_2_size);
delta_C2 = zeros(layer2_filter_num,conv_2_size,conv_2_size);
delta_C2_sig = zeros(layer2_filter_num,conv_2_size,conv_2_size);
delta_S1 = zeros(layer1_filter_num,pool_1_size,pool_1_size);
delta_C1 = zeros(layer1_filter_num,conv_1_size,conv_1_size);
delta_C1_sig = zeros(layer1_filter_num,conv_1_size,conv_1_size);

init_time = cputime;

input_image = squeeze(trInput(:,:,:,1));
input_label = trDes(1);
% y is a vector of labels
y_one_hot = zeros( num_class, 1 );
y_one_hot( input_label+1 ) = 1;

for l1_N=1:layer1_filter_num
    conv_1(l1_N,:,:) = conv2(input_image,squeeze(k1_layer(l1_N,:,:)), 'valid');
    sigmoid_1(l1_N,:,:) = Sigmoid(conv_1(l1_N,:,:)+b1_layer(l1_N));
end
% max pooling
for l1_N=1:layer1_filter_num
    %tmp = conv2(squeeze(sigmoid_1(l1_N,:,:)),ones(2)/conv_1_size^2, 'valid');
    %S1_layer(l1_N,:,:) = tmp(1:2:end,1:2:end);
    tmp = squeeze(sigmoid_1(l1_N,:,:));
    for i = 1:pool_1_size
        max_val = -1;
        for j = 1:pool_1_size
            iv = (i-1)*2 + 1;
            jv = (j-1)*2 + 1;
            S1_layer(l1_N,i,j) = max(max(max(tmp(iv,jv),tmp(iv,jv+1)),tmp(iv+1,jv)),tmp(iv+1,jv+1));
        end
    end
end

for l2_N=1:layer2_filter_num
    tmp_sum = zeros(conv_2_size,conv_2_size);
    for l1_N=1:layer1_filter_num
        conv_2(l2_N,l1_N,:,:) = conv2(squeeze(S1_layer(l1_N,:,:)),squeeze(k2_layer(l1_N,l2_N,:,:)), 'valid');
        tmp_sum = tmp_sum + squeeze(conv_2(l2_N,l1_N,:,:));
    end
    sigmoid_2(l2_N,:,:) = Sigmoid(tmp_sum+b2_layer(l2_N));
end
% max pooling
for l2_N=1:layer2_filter_num
    %tmp = conv2(squeeze(sigmoid_2(l2_N,:,:)),ones(2)/conv_2_size^2, 'valid');
    %S2_layer(l2_N,:,:) = tmp(1:2:end,1:2:end);
    tmp = squeeze(sigmoid_2(l2_N,:,:));
    for i = 1:pool_2_size
        max_val = -1;
        for j = 1:pool_2_size
            iv = (i-1)*2 + 1;
            jv = (j-1)*2 + 1;
            S2_layer(l1_N,i,j) = max(max(max(tmp(iv,jv),tmp(iv,jv+1)),tmp(iv+1,jv)),tmp(iv+1,jv+1));
        end
    end
end

%vectorization
fv = [];
for l2_N=1:layer2_filter_num
    sa = size(S2_layer(l2_N,:,:));
    fv = [fv; reshape(S2_layer(l2_N,:,:), sa(2)*sa(3), sa(1))];
end

%fully conected layer
FC_layer = Sigmoid(FC_W*fv + FC_b);
Loss_list = (FC_layer - y_one_hot);

FC_layer
y_one_hot
% back propagation
delta_y = (FC_layer - y_one_hot).*deSigmoid(FC_layer);
delta_W = delta_y.* (fv.');
delta_B = delta_y;
delta_f = FC_W.' * delta_y;
%reshape to square
p2_square = pool_2_size * pool_2_size;
for l2_N=1:layer2_filter_num
    delta_S2(l2_N,:,:) = reshape(delta_f((l2_N - 1)*p2_square+1 : l2_N*p2_square,:),pool_2_size,pool_2_size);
end
%pooling recover
for l2_N=1:layer2_filter_num
    for i=1:conv_2_size
        for j=1:conv_2_size
            delta_C2(l2_N,i,j) = (1/4)*delta_S2(ceil(i/2),ceil(j/2));
        end
    end
end

for l2_N=1:layer2_filter_num
    delta_C2_sig(l2_N,:,:) = delta_C2(l2_N,:,:).* deSigmoid(sigmoid_2(l2_N,:,:));
end
for p=1:layer1_filter_num
    % z = zeros(filter_size,filter_size);
    for q=1:layer2_filter_num
        delta_k2(p,q,:,:) = conv2(squeeze(delta_C2_sig(q,:,:)),rot180(squeeze(S2_layer(p,:,:))),'valid');
    end
end
for q = 1:layer2_filter_num
    delta_b2(q) = sum(delta_C2_sig(q,:,:),'all');
end

for p = 1:layer1_filter_num
    z = zeros(pool_1_size,pool_1_size);
    for q = 1:layer2_filter_num
        z = z + conv2(squeeze(delta_C2_sig(q,:,:)),squeeze((rot180(k2_layer(p,q,:,:)))));
    end
    delta_S1(p,:,:) = z;
end

for p = 1:layer1_filter_num
    for i=1:conv_1_size
        for j=1:conv_1_size
            delta_C1(p,i,j) = (1/4)*delta_S1(p,ceil(i/2),ceil(j/2));
        end
    end
end

for p = 1:layer1_filter_num
    delta_C1_sig(p,:,:) = delta_C1(p,:,:).*deSigmoid(sigmoid_1(p,:,:));
end

for p = 1:layer1_filter_num
    delta_k1(p,:,:) = conv2(rot180(squeeze(input_image)),squeeze(delta_C1_sig(p,:,:)),'valid');
end

for p = 1:layer1_filter_num
    delta_b1(p) = sum(delta_C1_sig(p,:,:),'all');
end

% parameter update
for p = 1:layer1_filter_num
    k1_layer(p,:,:) = k1_layer(p,:,:) - learning_rate * delta_k1(p,:,:);
    b1_layer(p) = b1_layer(p) - learning_rate * delta_b1(p);
end

for q = 1:layer2_filter_num
    for p = 1:layer1_filter_num
        k2_layer(p,q,:,:) = k2_layer(p,q,:,:) - learning_rate * delta_k2(p,q,:,:);
    end
    b2_layer(q) = b2_layer(q) - learning_rate* delta_b2(q);
end

for n = 1:num_class
    FC_W(n) = FC_W(n) - learning_rate * delta_W(n);
    FC_b(n) = FC_b(n) - learning_rate * delta_B(n);
end

fig_conv(1) = figure(1)
act1 = squeeze(permute(conv_1,[2 3 1]));
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)

fig_conv(2) = figure(2)
act1 = permute(sigmoid_1,[2 3 1]);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)

fig_conv(3) = figure(3)
act1 = permute(S1_layer,[2 3 1]);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)

fig_conv(4) = figure(4)
tmp = (squeeze(conv_2(:,1,:,:)));
act1 = permute(tmp,[2 3 1]);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)

fig_conv(5) = figure(5)
tmp = (squeeze(sigmoid_2(:,1,:,:)));
act1 = permute(tmp,[2 3 1]);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)

fig_conv(6) = figure(6)
act1 = permute(S2_layer,[2 3 1]);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)

fig_conv(7) = figure(7)
act1 = reshape(FC_layer,1,1,10);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)
