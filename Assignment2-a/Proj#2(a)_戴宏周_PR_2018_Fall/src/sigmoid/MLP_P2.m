 
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

itermax=20000;
eta=0.09;            % (n -> eta -> learning rate)
beta=0.09;           % momentum term
Lowerlimit=0.001;
layer = [3 3];
title_text = sprintf('P2 BIJK: 2 X %d X %d X 2 \n iter = %d, eta = %f,beta = %f',layer(1),layer(2),itermax,eta,beta);
file_text = sprintf('P2_BIJK_2X%dX%dX2_iter_%d_eta_%f_beta_%f',layer(1),layer(2),itermax,eta,beta);

[wkj,wji,wib,error_r,ite] = train_BIJK_net(data,eta,beta,layer,2,2,itermax,Lowerlimit);

fig_error = figure(1);
hold on;
plot(ite, error_r);
title(title_text);
xlabel('iteration');
ylabel('error');
saveas(fig_error,strcat(file_text,'_error.jpg'));
saveas(fig_error,strcat(file_text,'_error.fig'));

fig_decision = figure(2);
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
        ob=[dx dy 1]';

        ok = FeedFoward_BIJK(wib,wji,wkj,ob,2,2,layer); 
        % Real output
        if ok(1)<0.5
            plot(dx,dy, 'k .');
        elseif ok(1)>0.5
            plot(dx,dy, 'r .');
        end
    end
end

title(title_text);
xlabel('iteration');
ylabel('error');

saveas(fig_decision,strcat(file_text,'_decision.jpg'));
saveas(fig_decision,strcat(file_text,'_decision.fig'));
