clear all;

x_train = Get_MNIST('train-images-idx3-ubyte');
y_train = Get_MNISTLABEL('train-labels-idx1-ubyte');
x_test = Get_MNIST('t10k-images-idx3-ubyte');
y_test = Get_MNISTLABEL('t10k-labels-idx1-ubyte');

x_train_list = x_train.';

train_size = size(y_train);
train_size = train_size(1);

x_train_list = x_train.';
y_train_list = int32(zeros(train_size,10));

for ix=1:1:train_size
    y_train_list(ix,y_train(ix,1)+1) = 1;
end

x_input = x_train_list.';
y_output = y_train_list.';

layer =[20];

net = feedforwardnet([layer]);
lr = 0.0001;
epochs = 1000;
mc = 0.9;
title_text = sprintf('Pacakge: 784 X %d X 10 \n epochs = %d, lr = %f, mc %f',layer(1),epochs,lr,mc);
file_text = sprintf('Package_784X%dX10_epochs_%d_lr_%f_mc_%f',layer(1),epochs,lr,mc);

net.trainParam.lr = lr;
net.trainParam.epochs = epochs;
net.trainParam.goal = 0;
net.trainParam.mc = mc;
net.divideFcn= 'dividerand';
net.divideParam.trainRatio= 1;
net.divideParam.valRatio= 0;
net.divideParam.testRatio=0;
net.trainFcn = 'traingdx';
net.performParam.normalization = 'standard';
net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'logsig';
net.output.processFcns = {'mapminmax'};
net.input.processFcns = {'mapminmax'};
net = train(net,x_input,y_output);

view(net);

x_test_list = x_test.';
result_ok=zeros(10,10000);

%for ix=1:1:10000
%    oi = single([x_test_list(ix,1:784).']);
%    ok = net(oi);
%    result_ok(:,ix) = ok;
%    [M,I] = max(ok);
%    result_r(ix) = I-1;
%end

y_pred_label = sim(net,x_test_list.');

for ix=1:1:10000
    [M,I] = max(y_pred_label(:,ix));
    result_r(ix) = I - 1;
end

fig_train = figure(1)                                          % plot images
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

fig_test = figure(2)                       % plot images
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

saveas(fig_train,strcat(file_text,'_test.jpg'));
saveas(fig_train,strcat(file_text,'_test.fig'));
fig_confu = figure(3)
set(fig_confu, 'Position', get(0, 'Screensize'));
Confu = confusionmat(single(y_test), single(result_r.'));
confusionchart(single(y_test), single(result_r.'),'Title','Package Confusion Matrix', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
saveas(fig_confu,strcat(file_text,'_conf.jpg'));
saveas(fig_confu,strcat(file_text,'_conf.fig'));

%for ix=1:1:60000
%    oi = single([x_train_list(ix,1:784)].');
%    ok = net(oi);
%    [M,I] = max(ok);
%    result_tr(ix) = I-1;
%end
%
%fig_confu2 = figure(6)
%set(fig_confu2, 'Position', get(0, 'Screensize'));
%Confu = confusionmat(single(y_train), single(result_tr.'));
%confusionchart(single(y_train), single(result_tr.'),'Title','Sigmoid Confusion Matrix', ...
%    'RowSummary','row-normalized', ...
%    'ColumnSummary','column-normalized');
%saveas(fig_confu2,strcat(file_text,'_conf_train.jpg'));
%saveas(fig_confu2,strcat(file_text,'_conf_train.fig'));
trace(Confu)
