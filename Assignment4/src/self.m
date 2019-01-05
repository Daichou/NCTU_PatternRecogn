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

learning_rate = 0.2;

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
k1_layer = (rand(layer1_filter_num,filter_size,filter_size)-0.5)*2*sqrt(6/((1+6)*25));
k2_layer = (rand(layer1_filter_num,layer2_filter_num,filter_size,filter_size)-0.5)*2*sqrt(6/(18*25));
FC_W = (rand(num_class,pooling2_size)-0.5)*2*sqrt(6/(192+10));
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

for iter=1:10
    Loss = 0;
    Correctness = 0;
    for number_of_input=1:1000
        input_image = squeeze(trInput(:,:,:,number_of_input));
        input_label = trDes(number_of_input);
        % y is a vector of labels
        y_one_hot = zeros( num_class, 1 );
        y_one_hot( input_label+1 ) = 1;

		for p=1:layer1_filter_num
            conv_1(p,:,:) = conv2(input_image,squeeze(k1_layer(p,:,:)), 'valid');
            sigmoid_1(p,:,:) = Sigmoid(conv_1(p,:,:)+b1_layer(p));
        end
        % max pooling
        for p=1:layer1_filter_num
            %tmp = conv2(squeeze(sigmoid_1(p,:,:)),ones(2)/conv_1_size^2, 'valid');
            %S1_layer(p,:,:) = tmp(1:2:end,1:2:end);
            tmp = squeeze(sigmoid_1(p,:,:));
            for i = 1:pool_1_size
                max_val = -1;
                for j = 1:pool_1_size
                    iv = (i-1)*2 + 1;
                    jv = (j-1)*2 + 1;
                    %S1_layer(p,i,j) = max(max(max(tmp(iv,jv),tmp(iv,jv+1)),tmp(iv+1,jv)),tmp(iv+1,jv+1));
                    S1_layer(p,i,j) = (tmp(iv,jv) + tmp(iv,jv+1) + tmp(iv+1,jv) + tmp(iv+1,jv+1))/4;
                end
            end
        end

        for q=1:layer2_filter_num
            tmp_sum = zeros(conv_2_size,conv_2_size);
            for p=1:layer1_filter_num
                conv_2(q,p,:,:) = conv2(squeeze(S1_layer(p,:,:)),squeeze(k2_layer(p,q,:,:)), 'valid');
                tmp_sum = tmp_sum + squeeze(conv_2(q,p,:,:));
            end
            sigmoid_2(q,:,:) = Sigmoid(tmp_sum+b2_layer(q));
        end
        % max pooling
        for q=1:layer2_filter_num
            %tmp = conv2(squeeze(sigmoid_2(l2_N,:,:)),ones(2)/conv_2_size^2, 'valid');
            %S2_layer(l2_N,:,:) = tmp(1:2:end,1:2:end);
            tmp = squeeze(sigmoid_2(q,:,:));
            for i = 1:pool_2_size
                max_val = -1;
                for j = 1:pool_2_size
                    iv = (i-1)*2 + 1;
                    jv = (j-1)*2 + 1;
                    %S2_layer(q,i,j) = max(max(max(tmp(iv,jv),tmp(iv,jv+1)),tmp(iv+1,jv)),tmp(iv+1,jv+1));
                    S2_layer(q,i,j) = (tmp(iv,jv) + tmp(iv,jv+1) + tmp(iv+1,jv) + tmp(iv+1,jv+1))/4;
                end
            end
        end

        %vectorization
        fv = [];
        for q=1:layer2_filter_num
            sa = size(S2_layer(q,:,:));
            fv = [fv; reshape(S2_layer(q,:,:), sa(2)*sa(3), sa(1))];
        end
        %fully conected layer
        FC_layer = Sigmoid(FC_W*fv + FC_b);
        Loss_list = (FC_layer - y_one_hot);

        Loss = Loss + (FC_layer - y_one_hot)'*(FC_layer - y_one_hot)/2;
        %FC_layer
        %y_one_hot
        [M,I] = max(FC_layer);
        if (I-1 == input_label)
            Correctness = Correctness + 1;
        end
        % back propagation
        delta_y = (FC_layer - y_one_hot).*deSigmoid(FC_layer);
        delta_W = delta_y.* (fv.');
        delta_B = delta_y;
        delta_f = FC_W.' * delta_y;
        %reshape to square
        p2_square = pool_2_size * pool_2_size;
        for q=1:layer2_filter_num
            delta_S2(q,:,:) = reshape(delta_f((q - 1)*p2_square+1 : q*p2_square),pool_2_size,pool_2_size);
        end
        %pooling recover
        for q=1:layer2_filter_num
            for i=1:conv_2_size
                for j=1:conv_2_size
                    delta_C2(q,i,j) = (1/4)*delta_S2(ceil(i/2),ceil(j/2));
                end
            end
        end

        for q=1:layer2_filter_num
            delta_C2_sig(q,:,:) = delta_C2(q,:,:).* deSigmoid(sigmoid_2(q,:,:));
        end
        for p=1:layer1_filter_num
            % z = zeros(filter_size,filter_size);
            for q=1:layer2_filter_num
                delta_k2(p,q,:,:) = conv2(rot180(squeeze(S1_layer(p,:,:))),squeeze(delta_C2_sig(q,:,:)),'valid');
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
        k1_layer = k1_layer - learning_rate * delta_k1;
        b1_layer = b1_layer -  learning_rate * delta_b1;

        k2_layer = k2_layer - learning_rate * delta_k2;
        b2_layer = b2_layer - learning_rate * delta_b2;

        FC_W = FC_W - learning_rate * delta_W;
        FC_b = FC_b - learning_rate * delta_B;
    end
    ite(iter) = iter;
    Loss_r(iter) = Loss/1000;
    acc_r(iter) = Correctness/1000;
    fprintf("Epochs: %d, Loss = %f, acc = %f\n",iter,Loss/1000,acc_r(iter))
    time_r(iter) = cputime-init_time;
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

