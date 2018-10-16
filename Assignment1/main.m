% HW1 p1

x = [-3:.1:9]; % range
norm = normpdf(x,3,2); % range, avg, stddev
figure;
plot(x,norm);
title('P11-d Gaussian function ');

% HW1 p2
figure;
mu = [1 2];
Sigma = [1 0; 0 1]; % covariance
x1 = -2:.1:4; x2 = -1:.1:5; % (min, step , max)
[X1,X2] = meshgrid(x1,x2); 
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
surf(x1,x2,F);
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
axis([-2 4 -1 5 0 .2]);
xlabel('x1'); ylabel('x2'); zlabel('Probability Density');
title('P2  2-d Gaussian function');

% HW1 p3
rng default;
r= normrnd(0,1,10000,1); % mu, sigma, x = 100000 y = 1 (2d array)
figure;
histogram(r);
title('P3 1-d Gaussian random data');

% HW1 p4
mu = [1 2];
sigma = [3 0; 0 4];
rng default;  % For reproducibility
R = mvnrnd(mu,sigma,10000);
figure;
plot(R(:,1),R(:,2),'.');
title('P4 2-d Gaussian random data');
figure;
hist3([R(:,1),R(:,2)],'CDataMode','auto','FaceColor','interp','Ctrs', {-8:0.2:10 -10:0.2:14} );
title('P4 2-d Gaussian random data');
[n,z] = hist3([R(:,1),R(:,2)]);
%HW1 p5
figure;
contour(z{1},z{2},n);
title('P5 2-d Gaussian random data contour');

%HW1 p6
figure;
fplot(@(x) x-1)
title('P6 y=x-1');

%HW1 p7
figure;
ezplot('x^2 + y^2 - 1')
title('P7 x^2+y^2=1');

%HW1 p8
figure;
ezplot('x^2+y^2/4-1')
title('P8 x^2+y^2/4=1');

%HW1 p9
figure;
ezplot('x^2-y^2/4-1')
title('P9 x^2-y^2/4=1');

%HW1 p10
figure;
x=0:0.1:10;
y = 2*x;
plot(x,y,'.');
title('P10 2x-y=0')

%HW1 p11
figure;
x = -2:0.08:2;
py = sqrt(4-x.^2);
ny = -1* sqrt(4-x.^2);
plot([x x],[py ny],'.k');
title('P11 x^2+y^2=4');

%HW1 P12
figure;
x = -2:0.08:2;
plot(x,sqrt(1-x.^2/4),'b.');
hold on;
plot(x,-1*sqrt(1-x.^2/4),'b.');
hold off;
title('P12 x^2/4+y^2=1');

%HW1 P13
figure;
x = -1:0.02:1;
y = 1./x;
plot(x,y,'b.');
title('P13 xy = 1');

%HW1 P14
figure;
i = 0:1:96;
theta = i.*pi/16;
r = 6.5*(104-i)/104;
plot(r.*sin(theta),r.*cos(theta),'ro');
hold on;
plot(-1*r.*sin(theta),-1*r.*cos(theta),'b*');
hold off;
title('P14');
