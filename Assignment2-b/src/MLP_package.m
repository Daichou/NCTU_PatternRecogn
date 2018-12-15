clear all;

x_train = Get_MNIST('train-images-idx3-ubyte');
y_train = uint8(Get_MNISTLABEL('train-labels-idx1-ubyte'));
x_test = Get_MNIST('t10k-images-idx3-ubyte');
y_test = uint8(Get_MNISTLABEL('t10k-labels-idx1-ubyte'));

x_train_list = x_train.';
y_train_list = y_train.';
net = feedforwardnet([3]);
net.trainParam.lr = 0.2;
net.trainParam.epochs = 10000;
net.trainParam.goal = 0.001;
net.divideFcn= 'dividerand';
net.divideParam.trainRatio= 1;
net.divideParam.valRatio= 0;
net.divideParam.testRatio=0;
net = train(net,x_train,y_train_list);
view(net);

