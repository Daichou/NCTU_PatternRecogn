clear all;
N=97;
i = 0:1:96;
theta = i.*pi/16;
r = 6.5*(104-i)/104;

data(1:97,1) = r.*sin(theta);
data(1:97,2) = r.*cos(theta);
data(1:97,3) = 0;
data(1:97,4) = 0;
data(1:97,5) = 1;
data(1:97,6) = 0;

data(98:98+96,1) = -1*r.*sin(theta);
data(98:98+96,2) = -1*r.*cos(theta);
data(98:98+96,3) = 0;
data(98:98+96,4) = 0;
data(98:98+96,5) = 0;
data(98:98+96,6) = 1;

x_max = max(data(:,1));
x_min = min(data(:,1));
y_max = max(data(:,2));
y_min = min(data(:,2));

n_data(:,1) = data(:,1);
n_data(:,2) = data(:,2);
data(:,1) = (data(:,1) - x_min)/(x_max - x_min);
data(:,2) = (data(:,2) - y_min)/(y_max - y_min);
data(:,3) = data(:,1).*data(:,1)+data(:,2).*data(:,2);
data(:,4) = data(:,1).*data(:,2);


layer = [40 40 40];
itermax = 40000;
eta = 0.00011;
beta = 0.00010;
Lowerlimit = 0.001;
title_text = sprintf('ABIJK: 3 X %d X %d X %d X 2 \n iter = %d, eta = %f, beta = %f', layer(1),layer(2),layer(3),itermax,eta,beta);
file_text = sprintf('P1_ABIJK_3X%dX%dX%dX2_iter_%d_eta_%f_beta_%f', layer(1),layer(2),layer(3),itermax,eta,beta);

[wkj,wji,wib,wba,error_r,ite] = train_ABIJK_net(data,eta,beta,layer,4,2,itermax,Lowerlimit);

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
    plot(n_data(n,1), n_data(n,2),'r o');
end
for n=N+1:1:N*2
    plot(n_data(n,1), n_data(n,2),'k s');
end
 
for ix=-30:1:31
    for iy=-35:1:36
        dx=0.2*(ix-1); 
        dy=0.2*(iy-1);
        ndx = (dx-x_min)/(x_max - x_min);
        ndy = (dy-y_min)/(y_max - y_min);
        oa=[ndx ndy ndx*ndx+ndy*ndy ndx*ndy 1]';

        ok = FeedFoward_ABIJK(wba,wib,wji,wkj,oa,4,2,layer);
        
        % Real output
        if ok(1,1)<0.5
            plot(dx,dy, 'k .');
        elseif ok(1,1)>0.5
            plot(dx,dy, 'r .');
        end
    end
end
title(title_text);
xlabel('iteration');
ylabel('error');

saveas(fig_decision,strcat(file_text,'_decision.jpg'));
saveas(fig_decision,strcat(file_text,'_decision.fig'));
