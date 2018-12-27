
trInput = Get_MNIST('train-images-idx3-ubyte');
tsInput = Get_MNIST('t10k-images-idx3-ubyte');
trDes = Get_MNISTLABEL('train-labels-idx1-ubyte');
tsDes = Get_MNISTLABEL('t10k-labels-idx1-ubyte');
n_class = 10;

trInput = reshape(trInput,28,28,1,60000);
tsInput = reshape(tsInput,28,28,1,10000);
trT = categorical(trDes);
tsT = categorical(tsDes);


% design model
layers = [
    imageInputLayer([28 28 1],'Name','input')
    
    convolution2dLayer(3,8,'Padding','same','Name','conv_1')
    batchNormalizationLayer('Name','norm_1')
    reluLayer('Name','relu_1')
    
    maxPooling2dLayer(2,'Stride',2,'Name','Mpool_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','norm_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','Mpool_2')
    
    
    fullyConnectedLayer(10,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')];
lgraph = layerGraph(layers);
figure
plot(lgraph);

options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 128,...
    'InitialLearnRate', 0.1,...
    'ExecutionEnvironment', 'auto',...
    'Verbose',false, ...
    'Plots', 'training-progress');

rng('default');

net = trainNetwork(trInput,trT,layers,options);

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

% ConfMat
[Mout,Iout]=max(tsOut);%value,index
[Mdes,Ides]=max(tpDes);
ConfMat=zeros(n_class,n_class);
c=0;%check how many wrong
for i=1:size(tsDes,1)
    if(Ides(i)~=Iout(i))
        c=c+1;
    end
    % ConfMat(i,j) num of should be i, but be j

    ConfMat(Ides(i),Iout(i))=ConfMat(Ides(i),Iout(i))+1;
end

figure

plotconfusion(tpDes,tsOut)








