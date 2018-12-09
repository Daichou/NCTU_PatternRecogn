clear all;

mu1 = [0 0];
sigma1 = [1 0; 0 1];
rng default;  % For reproducibility
R = mvnrnd(mu1,sigma1,150);
N = 150;
data(1:N,1) = R(:,1);
data(1:N,2) = R(:,2);
data(1:N,3) = 1;
data(1:N,4) = 0;
data(1:N,5) = 0;
data(1:N,6) = 0;

mu2 = [14 0];
sigma2 = [1 0; 0 4];
R = mvnrnd(mu2,sigma2,150);
data(N+1:2*N,1) = R(:,1);
data(N+1:2*N,2) = R(:,2);
data(N+1:2*N,3) = 0;
data(N+1:2*N,4) = 1;
data(N+1:2*N,5) = 0;
data(N+1:2*N,6) = 0;

mu3 = [7 14];
sigma3 = [4 0; 0 1];
R = mvnrnd(mu3,sigma3,150);
data(2*N+1:3*N,1) = R(:,1);
data(2*N+1:3*N,2) = R(:,2);
data(2*N+1:3*N,3) = 0;
data(2*N+1:3*N,4) = 0;
data(2*N+1:3*N,5) = 1;
data(2*N+1:3*N,6) = 0;

mu4 = [7 7];
sigma4 = [1 0; 0 1];
R = mvnrnd(mu4,sigma4,150);
data(3*N+1:4*N,1) = R(:,1);
data(3*N+1:4*N,2) = R(:,2);
data(3*N+1:4*N,3) = 0;
data(3*N+1:4*N,4) = 0;
data(3*N+1:4*N,5) = 0;
data(3*N+1:4*N,6) = 1;

x_input = [data(:,1) data(:,2)];
y_output = [data(:,3) data(:,4) data(:,5) data(:,6)];
x_input = x_input.';
y_output = y_output.';
net = feedforwardnet([3]);
net.trainParam.lr = 0.2;
net.trainParam.epochs = 10000;
net.trainParam.goal = 0.001;
net.divideFcn= 'dividerand';
net.divideParam.trainRatio= 1;
net.divideParam.valRatio= 0;
net.divideParam.testRatio=0;
net = train(net,x_input,y_output);
view(net);

title_text = sprintf('P3: 2X3X2\n lr = %d, epochs = %f', net.trainParam.lr, net.trainParam.epochs);
file_text = sprintf('P3_2X3X2_lr_%d_epochs_%f', net.trainParam.lr, net.trainParam.epochs);

fig_decision = figure(1);
hold on;

plot(data(1:N,1),data(1:N,2),'ro');
plot(data(N+1:2*N,1),data(N+1:2*N,2),'bo');
plot(data(2*N+1:3*N,1),data(2*N+1:3*N,2),'go');
plot(data(3*N+1:4*N,1),data(3*N+1:4*N,2),'ko');

for ix=-5*4:1:20*4
    for iy=-5*4:1:20*4
        dx=0.25*(ix-1);
        dy=0.25*(iy-1);

        final_out = net([dx dy].');

        % Real output
        [M,I] = max(final_out);
        if (I == 1)
            plot(dx,dy, 'r .');
        elseif (I == 2)
            plot(dx,dy, 'b .');
        elseif (I == 3)
            plot(dx,dy, 'g .');
        elseif (I == 4)
            plot(dx,dy, 'k .');
        end
    end
end
title(title_text);
xlabel('iteration');
ylabel('error');



saveas(fig_decision,strcat(file_text,'_decision.jpg'));
saveas(fig_decision,strcat(file_text,'_decision.fig'));
