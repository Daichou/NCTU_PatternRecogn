
clear all;

N=250;
theta1 = linspace(-180,180, N)*pi/360;
r = 8

data(1:N,1) = -5 + r*sin(theta1)+randn(1,N);
data(1:N,2) = r*cos(theta1)+randn(1,N);
data(1:N,3) = 1;
data(1:97,4) = 0;

data(N+1:2*N,1) = 5 + r*sin(theta1)+randn(1,N);
data(N+1:2*N,2) = -r*cos(theta1)+randn(1,N);
data(N+1:2*N,3) = 0;
data(N+1:2*N,4) = 1;

x_input = [data(:,1) data(:,2)];
y_output = [data(:,3) data(:,4)];
x_input = x_input.';
y_output = y_output.';
net = feedforwardnet([2 2]);
net.trainParam.lr = 0.2;
net.trainParam.epochs = 10000;
net.trainParam.goal = 0.001;
net = train(net,x_input,y_output);
view(net);

title_text = sprintf('P2: 2X2X2X2\n lr = %d, epochs = %f', net.trainParam.lr, net.trainParam.epochs);
file_text = sprintf('P2_2X2X2X2_lr_%d_epochs_%f', net.trainParam.lr, net.trainParam.epochs);

fig_decision = figure(1);
hold on;
for n=1:1:N
    plot(data(n,1), data(n,2),'r o');
end
for n=N+1:1:N*2
    plot(data(n,1), data(n,2),'k s');
end

for ix=-30:1:31
    for iy=-30:1:31
        dx=0.5*(ix-1);
        dy=0.5*(iy-1);

        final_out = net([dx dy].');

        % Real output
        if final_out(1)<0.5
            plot(dx,dy, 'k .');
        elseif final_out(1)>0.5
            plot(dx,dy, 'r .');
        end
    end
end
title(title_text);
xlabel('iteration');
ylabel('error');



saveas(fig_decision,strcat(file_text,'_decision.jpg'));
saveas(fig_decision,strcat(file_text,'_decision.fig'));
