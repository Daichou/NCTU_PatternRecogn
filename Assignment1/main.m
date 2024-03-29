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

%HW1 P15
figure;
x = [0 0 1 1 0 1 1 0];
y = [1 0 0 0 0 1 1 1];
z = [1 0 0 1 1 0 1 0];
classes = [1 2 2 2 1 2 1 1];
plot3(x(classes == 1),y(classes==1),z(classes==1),'bo');
hold on;
plot3(x(classes == 2),y(classes==2),z(classes==2),'rx');
grid on;
hold on;
x = 0:1:1;
y = 0:1:1;
[X,Y] = meshgrid(x,y);
Z = X - Y + 0.5;
surf(X,Y,Z)
hold off;

%HW1 P16
figure;
b_cen = [0,0];
b_l_radius = 10;
b_s_radius = 6;
b_theta = 1*pi*rand(1000,1)-0.5*pi;
b_r_radius = (b_l_radius - b_s_radius)*rand(1000,1) + b_s_radius;
b_x = b_r_radius.*sin(b_theta)+b_cen(1);
b_y = b_r_radius.*cos(b_theta)+b_cen(2);
plot(b_x,b_y,'b.')
hold on;
r_cen = [9,0];
r_l_radius = 10;
r_s_radius = 6;
r_theta = 1*pi*rand(1000,1)+0.5*pi;
r_r_radius = (r_l_radius - r_s_radius)*rand(1000,1) + r_s_radius;
r_x = r_r_radius.*sin(r_theta)+r_cen(1);
r_y = r_r_radius.*cos(r_theta)+r_cen(2);
plot(r_x,r_y,'r.')
hold off;
title('P16 2 moon')
%HW1 P17
figure;
x=0:0.01:pi;
for i = 1:5
    y_plot=sin(x*pi*i);
    plot(x,y_plot,'.k');
    axis([0,pi,-1.2,1.2]);
    title(strcat('P17 T/',num2str(i)))
    pause(5);
end
%HW1 P18
figure;
s1 = uint64(datenum(datetime)+cputime*1000)
s2 = uint64(datenum(datetime)+cputime*1000)
c = []
lower_bound = 0.0;
upper_bound = 1.0;
for i = 1:100
    x = uint64(s1);
    y = uint64(s2);
    s1 = y;
    x = bitxor(x,bitshift(x,32,'uint64'),'uint64');

    s2 = bitxor(bitxor(bitxor(x,y,'uint64'),bitshift(x,-17,'uint64'),'uint64'),bitshift(y,-26,'uint64'),'uint64');
    ans = double(s2) + double(y);
    if (ans > 2^64-1)
       ans = ans - 2^64+1;
    end
    bot = double(2^64-1);
    final = double(ans/bot);
    c = cat(1,c,final);
end
c = c.*(upper_bound-lower_bound)+lower_bound;
histogram(c,100);
title('P18 uniform random ditribution')

%HW1 P19
figure;
s1 = uint64(datenum(datetime)+cputime*1000);
s2 = uint64(datenum(datetime)+cputime*1000);
mean = 0;
std = 1;
normals = []
for j = 1:10000
    c = [];
    for i = 1:2
        x = uint64(s1);
        y = uint64(s2);
        s1 = y;
        x = bitxor(x,bitshift(x,32,'uint64'),'uint64');

        s2 = bitxor(bitxor(bitxor(x,y,'uint64'),bitshift(x,-17,'uint64'),'uint64'),bitshift(y,-26,'uint64'),'uint64');
        ans = double(s2) + double(y);
        if (ans > 2^64-1)
           ans = ans - 2^64+1;
        end
        bot = double(2^64-1);
        final = double(ans/bot);
        c = cat(1,c,final);
    end
    n_final = sqrt(-2*log(c(1)))*cos(2*pi*c(2))* std + mean;
    normals = cat(1,normals,n_final);
end
histogram(normals,100);
title('P19 normal distribution')
h = chi2gof(normals)

% HW1_P20
figure;
% load data
filename = 'train-images-idx3-ubyte'
fp = fopen(filename, 'rb');
 
magic = fread(fp, 1, 'int32', 0, 'ieee-be')
numImages = fread(fp, 1, 'int32', 0, 'ieee-be')
numRows = fread(fp, 1, 'int32', 0, 'ieee-be')
numCols = fread(fp, 1, 'int32', 0, 'ieee-be')
 
images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
% rotate image
images = permute(images,[2 1 3]);
 
fclose(fp);

images = reshape(images, numCols*numRows, numImages);
images = uint8(images)/255 ;

num=randi(length(images),10,15)

for i=0:9
    for j=0:14
        img(numCols*i+1:numCols*(i+1),numRows*j+1:numRows*(j+1))=reshape(images(:,num(i+1,j+1)),numCols,numRows);
    end
end

imshow(img)
