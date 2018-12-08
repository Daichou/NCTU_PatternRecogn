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

figure;
hold on;
plot(data(1:N,1),data(1:N,2),'ro');
plot(data(N+1:2*N,1),data(N+1:2*N,2),'bo');
plot(data(2*N+1:3*N,1),data(2*N+1:3*N,2),'go');
plot(data(3*N+1:4*N,1),data(3*N+1:4*N,2),'ko');
title('P4 2-d Gaussian random data');
