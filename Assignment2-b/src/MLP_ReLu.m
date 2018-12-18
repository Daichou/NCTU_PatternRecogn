clear all;

x_train = Get_MNIST('train-images-idx3-ubyte');
y_train = uint8(Get_MNISTLABEL('train-labels-idx1-ubyte'));
x_test = Get_MNIST('t10k-images-idx3-ubyte');
y_test = uint8(Get_MNISTLABEL('t10k-labels-idx1-ubyte'));

train_size = size(y_train);
train_size = train_size(1);

x_train_list = x_train.';
y_train_list = zeros(train_size,10);

for ix=1:1:train_size
    y_train_list(ix,y_train(ix,1)+1) = 1;
end

data = cat(2,x_train_list,y_train_list);

test_size = size(y_train);
test_size = train_size(1);

x_test_list = x_test.';
y_test_list = zeros(test_size,10);

data = cat(2,x_train_list,y_train_list);
layer = [100];
n_input = 60000;
n_output = 10;
itermax = 100;
eta = 0.001;
beta = 0.0009;
Lowerlimit = 0.001;
method = 0; % 1 : Sigmoid 0 : ReLu
title_text = sprintf('ReLu IJK: %d X %d X %d \n iter = %d, eta = %f, beta = %f',n_input,layer(1),n_output,itermax,eta,beta);
file_text = sprintf('ReLu_IJK_%dX%dX%d_iter_%d_eta_%f_beta_%f',n_input,layer(1),n_output,itermax,eta,beta);

[wkj,wji,error_r,ite,time_r] = train_IJK_net(data,eta,beta,layer,784,10,itermax,Lowerlimit,method);


fig_error = figure(1);
hold on;
set(fig_error, 'Position', get(0, 'Screensize'));
plot(ite, error_r);
title(title_text);
xlabel('iteration');
ylabel('error');
saveas(fig_error,strcat(file_text,'_error.jpg'));
saveas(fig_error,strcat(file_text,'_error.fig'));

fig_time = figure(2);
hold on;
set(fig_time, 'Position', get(0, 'Screensize'));

plot(time_r, error_r);
title(title_text);
xlabel('time');
ylabel('error');
saveas(fig_time,strcat(file_text,'_terror.jpg'));
saveas(fig_time,strcat(file_text,'_terror.fig'));

for ix=1:1:10000
    oi = single([x_test_list(ix,1:784) 1].');
    ok = FeedFoward_IJK(wji,wkj,oi,784,10,layer,method);
    [M,I] = max(ok);
    result_r(ix) = I-1;
end


fig_train = figure(3)                                          % plot images
set(fig_train, 'Position', get(0, 'Screensize'));
colormap(gray)                                  % set to grayscale
for i = 1:100
    subplot(10,10,i)
    digit = reshape(x_train(:,i), 28,28)';    % row = 28 x 28 image
    digit = permute(digit,[2 1 3]);
    imagesc(digit)                              % show the image
end
saveas(fig_train,strcat(file_text,'_train.jpg'));
saveas(fig_train,strcat(file_text,'_train.fig'));

fig_test = figure(4)                       % plot images
set(fig_test, 'Position', get(0, 'Screensize'));
colormap(gray)                                  % set to grayscale
for i = 1:100
    subplot(10,10,i)
    digit = reshape(x_test(:,i), 28,28)';    % row = 28 x 28 image
    digit = permute(digit,[2 1 3]);
    imagesc(digit)                              % show the image
    if y_test(i,1) == result_r(i)
        title_str = sprintf('desire:%d,pred:%d,True',y_test(i,1),result_r(i));
    else
        title_str = sprintf('desire:%d,pred:%d,False',y_test(i,1),result_r(i));
    end
    title(title_str)                    % show the label
end

saveas(fig_test,strcat(file_text,'_test.jpg'));
saveas(fig_test,strcat(file_text,'_test.fig'));
fig_confu = figure(5)
set(fig_confu, 'Position', get(0, 'Screensize'));
Confu = confusionmat(single(y_test), single(result_r.'));
confusionchart(single(y_test), single(result_r.'),'Title','ReLu Confusion Matrix', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
saveas(fig_confu,strcat(file_text,'_conf.jpg'));
saveas(fig_confu,strcat(file_text,'_conf.fig'));

for ix=1:1:60000
    oi = single([x_train_list(ix,1:784) 1].');
    ok = FeedFoward_IJK(wji,wkj,oi,784,10,layer,method);
    [M,I] = max(ok);
    result_train(ix) = I-1;
end

fig_confu2 = figure(6)
set(fig_confu2, 'Position', get(0, 'Screensize'));
Confu = confusionmat(single(y_train), single(result_train.'));
confusionchart(single(y_train), single(result_train.'),'Title','ReLu Confusion Matrix', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
saveas(fig_confu2,strcat(file_text,'_conf_train.jpg'));
saveas(fig_confu2,strcat(file_text,'_conf_train.fig'));
