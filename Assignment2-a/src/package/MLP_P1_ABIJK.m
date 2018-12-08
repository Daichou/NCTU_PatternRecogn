clear all;
N=97;
i = 0:1:96;
theta = i.*pi/16;
r = 6.5*(104-i)/104;

data(1:97,1) = r.*sin(theta);
data(1:97,2) = r.*cos(theta);
data(1:97,3) = 1;
data(1:97,4) = 0;

data(98:98+96,1) = -1*r.*sin(theta);
data(98:98+96,2) = -1*r.*cos(theta);
data(98:98+96,3) = 0;
data(98:98+96,4) = 1;

x_input = [data(:,1) data(:,2)];
y_output = [data(:,3) data(:,4)];

net = feedforwardnet([20 20 20]);
net.trainParam.lr = 0.2;
net.trainParam.epochs = 300;
net.trainParam.goal = 0.01;
net = train(net,x_input,y_output);
view(net);


