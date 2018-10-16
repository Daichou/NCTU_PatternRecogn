% HW1 p1

x = [-3:.1:9]; % range
norm = normpdf(x,3,2); % range, avg, stddev
figure;
plot(x,norm)
title('P11-d Gaussian function ')

% HW1 p2
figure
mu = [1 2];
Sigma = [1 0; 0 1]; % covariance
x1 = -2:.1:4; x2 = -1:.1:5; % (min, step , max)
[X1,X2] = meshgrid(x1,x2); 
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([-2 4 -1 5 0 .2])
xlabel('x1'); ylabel('x2'); zlabel('Probability Density');
title('P2  2-d Gaussian function')

% HW1 p3
rng default;
r= normrnd(0,1,10000,1) % mu, sigma, x = 100000 y = 1 (2d array)
figure
histogram(r)
title('P3 1-d Gaussian random data')

% HW1 p4
mu = [1 2];
sigma = [3 0; 0 4];
rng default  % For reproducibility
R = mvnrnd(mu,sigma,10000);
figure
plot(R(:,1),R(:,2),'.')
figure
hist3([R(:,1),R(:,2)],'CDataMode','auto','FaceColor','interp','Ctrs', {-8:0.2:10 -10:0.2:14} )