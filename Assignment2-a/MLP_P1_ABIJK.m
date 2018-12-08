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
% A = 2+1 % B = 5+1; % I=3+1;% J=3+1;% K=2;
nvectors=N*2;
ninpdim_with_bias=3;

neuron_hid_layerJ=200;
neuron_hid_layerJ_with_bias=neuron_hid_layerJ+1;
neuron_hid_layerI=200;
neuron_hid_layerI_with_bias=neuron_hid_layerI+1;
neuron_hid_layerB=200;

neuron_hid_layerB_with_bias=neuron_hid_layerB+1;
noutdim=2;

%initialize
wkj = randn(noutdim,neuron_hid_layerI_with_bias);
wkj_tmp = zeros(size(wkj));
wji = randn(neuron_hid_layerJ_with_bias,neuron_hid_layerI_with_bias);
wib = randn(neuron_hid_layerI_with_bias,neuron_hid_layerB_with_bias);
wba = randn(neuron_hid_layerB_with_bias,ninpdim_with_bias);
olddelwkj=zeros(noutdim , neuron_hid_layerJ_with_bias); % weight of Wkj (J -> K)
olddelwji=zeros(neuron_hid_layerJ_with_bias , neuron_hid_layerI_with_bias);   % weight of Wji (I -> J)
olddelwib=zeros(neuron_hid_layerB_with_bias , neuron_hid_layerI_with_bias);   % weight of Wji (B -> I)
olddelwba=zeros(neuron_hid_layerB_with_bias , ninpdim_with_bias);   % weight of Wji (A -> B)
oa = zeros(ninpdim_with_bias,1);
oa(ninpdim_with_bias) = 1;

sb = zeros(neuron_hid_layerB_with_bias,1);
ob = zeros(neuron_hid_layerB_with_bias,1);
ob(neuron_hid_layerB_with_bias) = 1;       % output of data

si = zeros(ninpdim_with_bias,1);       % input of hidden layer i
oi = zeros(neuron_hid_layerJ_with_bias,1);
oi(neuron_hid_layerI_with_bias) = 1;    % output of hidden layer i

sj = zeros(neuron_hid_layerI_with_bias,1);      % input of hidden layer j
oj = zeros(neuron_hid_layerJ_with_bias,1);
oj(neuron_hid_layerJ_with_bias) = 1;    % output of hidden layer j

sk = zeros(neuron_hid_layerJ_with_bias,1);        % input of output layer k
ok = zeros(noutdim,1);        % net output
dk = zeros(noutdim,1);        % desired output

Lowerlimit=0.01;
itermax=100000;

eta=0.05;            % (n -> eta -> learning rate)
beta=0.05;           % momentum term

 
iter=0;
error_avg=10;

title_text = sprintf('ABIJK:%d X %d X %d X %d X %d \n iter = %d, eta = %f, beta',ninpdim_with_bias,neuron_hid_layerB,neuron_hid_layerI,neuron_hid_layerJ,noutdim,itermax,eta,beta);
file_text = sprintf('P1_ABIJK_%dX%dX%dX%dX%d_iter_%d_eta_%f_beta_%f',ninpdim_with_bias,neuron_hid_layerB,neuron_hid_layerI,neuron_hid_layerJ,noutdim,itermax,eta,beta);

% internal variables
deltak = zeros(1,noutdim);
deltaj = zeros(1,neuron_hid_layerJ_with_bias);
deltai = zeros(1,neuron_hid_layerI_with_bias);
deltab = zeros(1,neuron_hid_layerB_with_bias);
sumback = zeros(1,max(neuron_hid_layerJ_with_bias, max(neuron_hid_layerI_with_bias,neuron_hid_layerB_with_bias)));
 
while (error_avg > Lowerlimit) && (iter<itermax)
    iter=iter+1;
    error=0;
 
% Forward Computation:
    for ivector=1:nvectors
        oa=[data(ivector,1) data(ivector,2) 1]';
        dk=[data(ivector,3) data(ivector,4)]';

        for j=1:neuron_hid_layerB
            sb(j)=wba(j,:)*oa;
            ob(j)=1/(1+exp(-sb(j)));    % sigmoid
        end
        ob(neuron_hid_layerB_with_bias)=1.0;

        for j=1:neuron_hid_layerI
            si(j)=wib(j,:)*ob;
            oi(j)=1/(1+exp(-si(j)));    % sigmoid
        end
        oi(neuron_hid_layerI_with_bias)=1.0;

        for j=1:neuron_hid_layerJ
            sj(j)=wji(j,:)*oi;
            oj(j)=1/(1+exp(-sj(j)));    % sigmoid
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
            deltaj(j)=oj(j)*(1.0-oj(j))*sumback(j);
         end

         for j=1:neuron_hid_layerI
            sumback(j)=0.0;
            for k=1:neuron_hid_layerJ_with_bias
               sumback(j)=sumback(j)+deltaj(k)*wji(k,j);
            end
            deltai(j)=oi(j)*(1.0-oi(j))*sumback(j);
         end
 
         for j=1:neuron_hid_layerB
            sumback(j)=0.0;
            for k=1:neuron_hid_layerI_with_bias
               sumback(j)=sumback(j)+deltai(k)*wib(k,j);
            end
            deltab(j)=ob(j)*(1.0-ob(j))*sumback(j);
         end

         for i=1:ninpdim_with_bias
            for j=1:neuron_hid_layerB
               wba(j,i)=wba(j,i)+eta*deltab(j)*oa(i)+beta*olddelwba(j,i);
               olddelwba(j,i)=eta*deltab(j)*oa(i)+beta*olddelwba(j,i);
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
hold on;
for n=1:1:N
    plot(data(n,1), data(n,2),'r o');
end
for n=N+1:1:N*2
    plot(data(n,1), data(n,2),'k s');
end
 
for ix=-30:1:31
    for iy=-30:1:31
        dx=0.2*(ix-1); 
        dy=0.2*(iy-1);
        oa=[dx dy 1]';
 
        for j=1:neuron_hid_layerB
            sb(j)=wba(j,:)*oa;
            ob(j)=1/(1+exp(-sb(j)));    % sigmoid
        end
        ob(neuron_hid_layerB_with_bias)=1.0;

        for j=1:neuron_hid_layerI
            si(j)=wib(j,:)*ob;
            oi(j)=1/(1+exp(-si(j)));    % sigmoid
        end
        oi(neuron_hid_layerI_with_bias)=1.0;

        for j=1:neuron_hid_layerJ
            sj(j)=wji(j,:)*oi;
            oj(j)=1/(1+exp(-sj(j)));    % sigmoid
        end
        oj(neuron_hid_layerJ_with_bias)=1.0;
 
        for k=1:noutdim
            sk(k)=wkj(k,:)*oj;
            ok(k)=1/(1+exp(-sk(k)));    % signmoid
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
