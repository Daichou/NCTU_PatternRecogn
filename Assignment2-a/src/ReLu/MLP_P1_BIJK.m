clear all;

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

x_max = max(data(:,1));
x_min = min(data(:,1));
y_max = max(data(:,2));
y_min = min(data(:,2));

n_data(:,1) = (data(:,1) - x_min)/(x_max - x_min);
n_data(:,2) = (data(:,2) - y_min)/(y_max - y_min);
n_data(:,3) = n_data(:,1).*n_data(:,2);

% B = 2+1; % I=3+1;% J=3+1;% K=2;
nvectors=97+97;
ninpdim_with_bias=3;
neuron_hid_layerJ=200;
neuron_hid_layerJ_with_bias=neuron_hid_layerJ+1;
neuron_hid_layerI=200;
neuron_hid_layerI_with_bias=neuron_hid_layerI+1;
noutdim=2;

%initialize
wkj = randn(noutdim,neuron_hid_layerI_with_bias);
wkj_tmp = zeros(size(wkj));
wji = randn(neuron_hid_layerJ_with_bias,neuron_hid_layerI_with_bias);
wib = randn(neuron_hid_layerI_with_bias,ninpdim_with_bias);
olddelwkj=zeros(noutdim,neuron_hid_layerI_with_bias); % weight of Wkj (J -> K)
olddelwji=zeros(neuron_hid_layerI_with_bias,neuron_hid_layerJ_with_bias);   % weight of Wji (I -> J)
olddelwib=zeros(neuron_hid_layerJ_with_bias,ninpdim_with_bias);   % weight of Wji (I -> J)
ob = zeros(ninpdim_with_bias,1);
ob(ninpdim_with_bias) = 1;       % output of data
si = zeros(ninpdim_with_bias,1);       % input of hidden layer i
oi = zeros(neuron_hid_layerJ_with_bias,1);
oi(neuron_hid_layerI_with_bias) = 1;    % output of hidden layer i
sj = zeros(neuron_hid_layerI_with_bias,1);      % input of hidden layer j
oj = zeros(neuron_hid_layerJ_with_bias,1);
oj(neuron_hid_layerJ_with_bias) = 1;    % output of hidden layer j
sk = zeros(neuron_hid_layerJ_with_bias,1);        % input of output layer k
ok = zeros(noutdim,1);        % net output
dk = zeros(noutdim,1);        % desired output

Lowerlimit=0.02;
itermax=20000;
eta=0.001;            % (n -> eta -> learning rate)
beta=0.001;           % momentum term
 
iter=0;
error_avg=10;
 
 
title_text = sprintf('BIJK:%d X %d X %d X %d \n iter = %d, eta = %f',ninpdim_with_bias,neuron_hid_layerI,neuron_hid_layerJ,noutdim,itermax,eta);
file_text = sprintf('P1_BIJK_%dX%dX%dX%d_iter_%d_eta_%f',ninpdim_with_bias,neuron_hid_layerI,neuron_hid_layerJ,noutdim,itermax,eta);
% internal variables
deltak = zeros(1,noutdim);
deltaj = zeros(1,neuron_hid_layerJ_with_bias);
deltai = zeros(1,neuron_hid_layerI_with_bias);
sumback = zeros(1,max(neuron_hid_layerJ_with_bias,neuron_hid_layerI_with_bias));
 
while (error_avg > Lowerlimit) && (iter<itermax)
    iter=iter+1;
    error=0;
    data_index = randperm(length(data)); 
% Forward Computation:
    for ivector=1:nvectors
        r_index = data_index(ivector);
        ob=[n_data(r_index,1) n_data(r_index,2) 1]';
        dk=[data(r_index,3) data(r_index,4)]';

        for j=1:neuron_hid_layerI
            si(j)=wib(j,:)*ob;
            oi(j)=max(si(j),0);    % RelU
        end
        oi(neuron_hid_layerI_with_bias)=1.0;

        for j=1:neuron_hid_layerJ
            sj(j)=wji(j,:)*oi;
            oj(j)=max(sj(j),0);    % ReLU
        end
        oj(neuron_hid_layerJ_with_bias)=1.0;
 
        for k=1:noutdim
            sk(k)=wkj(k,:)*oj;
            ok(k)=1/(1+exp(-sk(k)));    % signmoid
        end
        
        %error=error+ (dk-ok)' *(dk-ok)/2;
        error=error+sum(abs(dk-ok)); % abs is absolute each element
        
% Backward learning:
 
         for k=1:noutdim
            deltak(k)=(dk(k)-ok(k))*ok(k)*(1.0-ok(k)); % gradient term
         end
 
         for j=1:neuron_hid_layerJ_with_bias
            for k=1:noutdim
               wkj_tmp(k,j)=wkj(k,j)+eta*deltak(k)*oj(j)+beta*olddelwkj(k,j);
               olddelwkj(k,j)=eta*deltak(k)*oj(j)+beta*olddelwkj(k,j);
            end
         end
 
         for j=1:neuron_hid_layerJ
            sumback(j)=0.0;
            for k=1:noutdim
               sumback(j)=sumback(j)+deltak(k)*wkj(k,j);
            end
            deltaj(j)=(oj(j) > 0)*sumback(j);
         end

         for j=1:neuron_hid_layerI
            sumback(j)=0.0;
            for k=1:neuron_hid_layerJ_with_bias
               sumback(j)=sumback(j)+deltaj(k)*wji(k,j);
            end
            deltai(j)=(oi(j) > 0)*sumback(j);
         end
 
         for i=1:ninpdim_with_bias
            for j=1:neuron_hid_layerI
               wib(j,i)=wib(j,i)+eta*deltai(j)*ob(i)+beta*olddelwib(j,i);
               olddelwib(j,i)=eta*deltai(j)*ob(i)+beta*olddelwib(j,i);
            end
         end

         wkj = wkj_tmp;

    end
 
    ite(iter)=iter;
    error_avg=error/nvectors;
    error_r(iter)=error_avg;
end
 
fig_error = figure(1);
hold on;
plot(ite, error_r);
title(title_text);
xlabel('iteration');
ylabel('error');
saveas(fig_error,strcat(file_text,'_error.jpg'));
saveas(fig_error,strcat(file_text,'_error.fig'));
fig_decision = figure(2);
figure;
hold on;
for n=1:1:97
    plot(data(n,1), data(n,2),'r o');
end
for n=98:1:194
    plot(data(n,1), data(n,2),'k s');
end
 
for ix=-30:1:31
    for iy=-30:1:31
        dx=0.2*(ix-1); 
        dy=0.2*(iy-1);
        ndx = (dx - x_min)/(x_max - x_min);
        ndy = (dy - y_min)/(y_max - y_min);
        ob=[ndx ndy 1]';
 
        for j=1:neuron_hid_layerI
            si(j)=wib(j,:)*ob;
            oi(j)=max(si(j));    % sigmoid
        end
        oi(neuron_hid_layerI_with_bias)=1.0;

        for j=1:neuron_hid_layerJ
            sj(j)=wji(j,:)*oi;
            oj(j)=max(sj(j),0);    % sigmoid
        end
        oj(neuron_hid_layerJ_with_bias)=1.0;
 
        for k=1:noutdim
            sk(k)=wkj(k,:)*oj;
            ok(k)=max(sk(k),0);    % signmoid
        end

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
