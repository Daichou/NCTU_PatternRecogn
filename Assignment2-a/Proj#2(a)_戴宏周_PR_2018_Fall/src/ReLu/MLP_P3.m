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

% MLP
Lowerlimit=0.02;
itermax=20000;
eta=0.2;            % (n -> eta -> learning rate)
beta=0.18;           % momentum term
layer=[4];

title_text = sprintf('P3 IJK: 2 X %d X 2 \n iter = %d, eta = %f, beta = %f',layer(1),itermax,eta,beta);
file_text = sprintf('P3_IJK_2X%dX2_iter_%d_eta_%f_beta_%f',layer(1),itermax,eta,beta);

[wkj,wji,error_r,ite] = train_IJK_net(data,eta,beta,layer,2,4,itermax,Lowerlimit);

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

plot(data(1:N,1),data(1:N,2),'ro');
plot(data(N+1:2*N,1),data(N+1:2*N,2),'bo');
plot(data(2*N+1:3*N,1),data(2*N+1:3*N,2),'go');
plot(data(3*N+1:4*N,1),data(3*N+1:4*N,2),'ko');

for ix=-5*4:1:20*4
    for iy=-5*4:1:20*4
        dx=0.25*(ix-1); 
        dy=0.25*(iy-1);
        oi=[dx dy 1]';
 
        ok = FeedFoward_IJK(wji,wkj,oi,2,4,layer);
        [M,I] = max(ok);
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

