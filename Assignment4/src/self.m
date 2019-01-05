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

layer1_filter_num = 6;
layer2_filter_num = 12;
filter_size = 5;
num_class = 10;
pooling2_size = layer2_filter_num*16;
image_size = 28;
conv_1_size = image_size+1-filter_size;
conv_2_size = image_size+2-filter_size*2;
pool_1_size = conv_1_size/2;
pool_2_size = conv_2_size/2;

b1_layer = zeros(layer1_filter_num,1);
b2_layer = zeros(layer2_filter_num,1);
k1_layer = rand(layer1_filter_num,filter_size,filter_size)*sqrt(layer1_filter_num/((1+layer1_filter_num)*filter_size*filter_size));
k2_layer = rand(layer1_filter_num,layer2_filter_num,filter_size,filter_size)*sqrt(layer1_filter_num/((layer2_filter_num+layer1_filter_num)*filter_size*filter_size));
FC_W = rand(num_class,pooling2_size)*sqrt(layer1_filter_num/(pooling2_size+num_class));
FC_b = zeros(num_class,1);
conv_1 = zeros(layer1_filter_num,conv_1_size,conv_1_size);
conv_2 = zeros(layer2_filter_num,conv_2_size,conv_2_size);
S1_layer = zeros(layer1_filter_num,pool_1_size,pool_1_size);
S2_layer = zeros(layer2_filter_num,pool_2_size,pool_2_size);

for number_of_input=1:1000
    input_image = squeeze(trInput(:,:,:,number_of_input));
    input_label = trDes(number_of_input);
    for l1_N=1:layer1_filter_num
        conv_1(l1_N,:,:) = Sigmoid(conv2(input_image,squeeze(k1_layer(l1_N,:,:)), 'valid')+b1_layer(l1_N));
    end
    % max pooling
    for l1_N=1:layer1_filter_num
        tmp = conv2(squeeze(conv_1(l1_N,:,:)),ones(2)/conv_1_size^2, 'valid');
        S1_layer(l1_N,:,:) = tmp(1:2:end,1:2:end);
    end

    for l2_N=1:layer2_filter_num
        tmp_sum = zeros(conv_2_size,conv_2_size);
        for l1_N=1:layer1_filter_num
            tmp_sum = tmp_sum + conv2(squeeze(conv_1(l1_N,:,:)),squeeze(k2_layer(l1_N,l2_N,:,:)), 'valid');
        end
        conv_2(l2_N,:,:) = Sigmoid(tmp_sum+b2_layer(l2_N));
    end
    % max pooling
    for l2_N=1:layer2_filter_num
        tmp = conv2(squeeze(conv_2(l2_N,:,:)),ones(2)/conv_2_size^2, 'valid');
        S2_layer(l2_N,:,:) = tmp(1:2:end,1:2:end);
    end

    %vectorization
    fv = [];
    for l2_N=1:layer2_filter_num
        sa = size(S2_layer(l2_N,:,:));
        fv = [fv; reshape(S2_layer(l2_N,:,:), sa(1)*sa(2), sa(3))];
    end
end

tsOutput= classify(net,tsInput);
tsOutput=tsOutput(1:numel(tsT));
accuracy = sum(tsOutput == tsT)/numel(tsT);
% categorical to matrix

tsOut=zeros(n_class,size(tsDes,1));
tpOut=grp2idx(tsOutput);
tpDes=zeros(n_class,size(tsDes,1));

for i=1:size(tsDes,1)
    tsOut(tpOut(i),i)=1;
    tpDes(tsDes(i)+1,i)=1;
end


fig_confu = figure(1)
set(fig_confu, 'Position', get(0, 'Screensize'));

plotconfusion(tpDes,tsOut)
saveas(fig_confu,strcat('CNN_confu_1000.jpg'));
saveas(fig_confu,strcat('CNN_confu_1000.fig'));

fig_predict = figure(2)
set(fig_predict, 'Position', get(0, 'Screensize'));
tickCell = {'XTickLabel',{},'YTickLabel',{},'XTick',{}, 'YTick',{}};

for i = 1:150
    subplot(15,10,i)
    digit = tsInput(:,:,:,i);    % row = 28 x 28 image
    %digit = permute(digit,[2 1 3]);
    imagesc(digit)                              % show the image
    [M,true_index] = max(tsOut(:,i));
    [M,pred_index] = max(tpDes(:,i));
    pred_index = pred_index - 1;
    true_index = true_index - 1;
    if  pred_index == true_index
        title_str = sprintf('des:%d,pred:%d,T',true_index,pred_index);
    else
        title_str = sprintf('des:%d,pred:%d,F',true_index,pred_index);
    end
    title(title_str,'FontSize',10)                    % show the label
    set(gca,tickCell{:});
end

saveas(fig_predict,strcat('CNN_predict_1000.jpg'));
saveas(fig_predict,strcat('CNN_predict_1000.fig'));


im = trInput(:,:,:,1);
imgSize = size(im);
imgSize = imgSize(1:2);

for i=1:1:12
    fig_conv(i+2) = figure(i+2)
    act1 = activations(net,im,net.Layers(i).Name);
    sz = size(act1);

    if (length(sz) < 3)
        act1ch32 = act1(:,:,1);
        act1ch32 = mat2gray(act1ch32);
        act1ch32 = imresize(act1ch32,imgSize);
        I = imtile({im,act1ch32});
    else
        act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
        I = imtile(imresize(mat2gray(act1),[48 48]));
    end
    imshow(I)
    name = net.Layers(i).Name;
    title_str = sprintf('layer%d_%s_Features',i,name);
    title(['Layer ',name,' Features'])
    saveas(fig_conv(i+2),strcat(title_str,'_1000.jpg'));
    saveas(fig_conv(i+2),strcat(title_str,'_1000.fig'));
    i = i +1;
end

