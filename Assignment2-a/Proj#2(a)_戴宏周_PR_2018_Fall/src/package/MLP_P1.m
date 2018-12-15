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

x_input = x_input.';
y_output = y_output.';

net = feedforwardnet([12 12 12]);
net.trainParam.lr = 0.01;
net.trainParam.epochs = 10000;
net.trainParam.goal = 0.0001;
net.divideFcn= 'dividerand';
net.divideParam.trainRatio= 1;
net.divideParam.valRatio= 0;
net.divideParam.testRatio=0;
net = train(net,x_input,y_output);
view(net);


title_text = sprintf('P1: 2X12X12X12X2\n lr = %d, epochs = %f', net.trainParam.lr, net.trainParam.epochs);
file_text = sprintf('P1_2X12X12X12X2_lr_%d_epochs_%f', net.trainParam.lr, net.trainParam.epochs);

fig_decision = figure(1);
hold on;
for n=1:1:97
    plot(data(n,1), data(n,2),'r o');
end
for n=98:1:194
    plot(data(n,1), data(n,2),'k s');
end
 
for ix=-7*5:1:7*5
    for iy=-10*5:1:10*5
        dx=0.2*(ix-1); 
        dy=0.2*(iy-1);

        final_out = net([dx dy].');

        % Real output
        if final_out(1)<0.5
            plot(dx,dy, 'k .');
        elseif final_out(1)>0.5
            plot(dx,dy, 'r .');
        end
    end
end

saveas(fig_decision,strcat(file_text,'_decision.jpg'));
saveas(fig_decision,strcat(file_text,'_decision.fig'));


