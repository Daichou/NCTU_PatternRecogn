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
    'MaxEpochs', 10, ...
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

