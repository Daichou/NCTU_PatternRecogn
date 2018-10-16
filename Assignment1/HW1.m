% HW1 p1

x = [-3:.1:9]; % range
norm = normpdf(x,3,2); % range, avg, stddev
figure;
plot(x,norm)

% HW1 p2
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