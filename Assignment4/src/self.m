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

learning_rate = 0.4;
iteration = 30;

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

for iter=1:iteration
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
    iter_r(iter) = iter;
    loss_r(iter) = Loss/1000;
    acc_r(iter) = Correctness/1000;
    fprintf("Epochs: %d, Loss = %f, acc = %f\n",iter,Loss/1000,acc_r(iter))
    time_r(iter) = cputime-init_time;
end

Correctness = 0;
for number_of_input=1:1000
    input_image = squeeze(tsInput(:,:,:,number_of_input));
    input_label = tsDes(number_of_input);
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

    [M,I] = max(FC_layer);
    if (I-1 == input_label)
        Correctness = Correctness + 1;
    end
    tsOutput(number_of_input) = I-1;
end

fig_confu = figure(1)
set(fig_confu, 'Position', get(0, 'Screensize'));

plotconfusion(categorical(tsDes),categorical(tsOutput.'))
saveas(fig_confu,strcat('CNN_confu_1000.jpg'));
saveas(fig_confu,strcat('CNN_confu_1000.fig'));

fig_predict = figure(2)
set(fig_predict, 'Position', get(0, 'Screensize'));
tickCell = {'XTickLabel',{},'YTickLabel',{},'XTick',{}, 'YTick',{}};

for i = 1:150
    subplot(15,10,i)
    digit = tsInput(:,:,:,i);    % row = 28 x 28 image
    %digit = permute(digit,[2 1 3]);
    imagesc(digit)
    true_index = tsDes(i);
    pred_index = tsOutput(i);
    % show the image
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



fig_conv(1) = figure(3)
act1 = squeeze(permute(conv_1,[2 3 1]));
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)
title_str = 'conv_1';
title(title_str,'FontSize',10)                    % show the label
saveas(fig_conv(1),strcat('conv_1layer.jpg'));
saveas(fig_conv(1),strcat('conv_1layer.fig'));

fig_conv(2) = figure(4)
act1 = permute(sigmoid_1,[2 3 1]);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)
title_str = 'sigmoid_1';
title(title_str,'FontSize',10)                    % show the label
saveas(fig_conv(2),strcat('sigmoid_1layer.jpg'));
saveas(fig_conv(2),strcat('sigmoid_1layer.fig'));

fig_conv(3) = figure(5)
act1 = permute(S1_layer,[2 3 1]);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)
title_str = 'pooling 1';
title(title_str,'FontSize',10)                    % show the label
saveas(fig_conv(3),strcat('pooling_1layer.jpg'));
saveas(fig_conv(3),strcat('pooling_1layer.fig'));

fig_conv(4) = figure(6)
tmp = (squeeze(conv_2(:,1,:,:)));
act1 = permute(tmp,[2 3 1]);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)
title_str = 'conv 2';
title(title_str,'FontSize',10)                    % show the label
saveas(fig_conv(4),strcat('conv_2_layer.jpg'));
saveas(fig_conv(4),strcat('conv_2_layer.fig'));

fig_conv(5) = figure(7)
tmp = (squeeze(sigmoid_2(:,1,:,:)));
act1 = permute(tmp,[2 3 1]);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)
title_str = 'sigmoid 2';
title(title_str,'FontSize',10)                    % show the label
saveas(fig_conv(5),strcat('sigmoid_2_layer.jpg'));
saveas(fig_conv(5),strcat('sigmoid_2_layer.fig'));

fig_conv(6) = figure(8)
act1 = permute(S2_layer,[2 3 1]);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)
title_str = 'pooling 2';
title(title_str,'FontSize',10)                    % show the label
saveas(fig_conv(6),strcat('pooling_2_layer.jpg'));
saveas(fig_conv(6),strcat('pooling_2_layer.fig'));

fig_conv(7) = figure(9)
act1 = reshape(FC_layer,1,1,10);
I = imtile(imresize(mat2gray(act1),[48 48]));
imshow(I)
title_str = 'FC layer';
title(title_str,'FontSize',10)                    % show the label
saveas(fig_conv(7),strcat('FC_layer.jpg'));
saveas(fig_conv(7),strcat('FC_layer.fig'));

fig_acc_iter = figure(10)
plot(iter_r,acc_r);
title('accuracy vs iteration', 'FontSize', 10)
saveas(fig_acc_iter,strcat('acc_iter.jpg'));
saveas(fig_acc_iter,strcat('acc_iter.fig'));

fig_loss_iter = figure(11)
plot(iter_r,loss_r);
title('loss vs iteration', 'FontSize', 10)
saveas(fig_loss_iter,strcat('loss_iter.jpg'));
saveas(fig_loss_iter,strcat('loss_iter.fig'));

fig_acc_time = figure(12)
plot(timee_r, acc_r);
title('accuracy vs cpu time', 'FontSize', 10)
saveas(fig_acc_time,strcat('acc_time.jpg'));
saveas(fig_acc_time,strcat('acc_time.fig'));

fig_loss_time = figure(13)
plot(time_r,loss_r);
title('loss vs cpu time', 'FontSize', 10)
saveas(fig_loss_time,strcat('loss_time.jpg'));
saveas(fig_loss_time,strcat('loss_time.fig'));
