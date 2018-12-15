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

layer = [200];
n_input = 60000;
n_output = 10;
itermax = 100000;
eta = 0.1;
beta = 0.09;
Lowerlimit = 0.001;
title_text = sprintf('Sigmoid ABIJK: %d X %d X %d \n iter = %d, eta = %f, beta = %f',n_input,layer(1),n_output,itermax,eta,beta);
file_text = sprintf('Sigmoid_ABIJK_%dX%dX%d_iter_%d_eta_%f_beta_%f',n_input,layer(1),n_output,itermax,eta,beta);

[wkj,wji,error_r,ite] = train_IJK_net(data,eta,beta,layer,784,10,itermax,Lowerlimit);


fig_error = figure(1);
hold on;

plot(ite, error_r);
title(title_text);
xlabel('iteration');
ylabel('error');
saveas(fig_error,strcat(file_text,'_error.jpg'));
saveas(fig_error,strcat(file_text,'_error.fig'));
