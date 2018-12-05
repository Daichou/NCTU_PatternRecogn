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
% I=2+1;% J=3+1;% K=2;
nvectors=N*4;
ninpdim1=3;
nhid=7;
nhid1=nhid+1;
noutdim=4;

%initialize
wkj = randn(noutdim,nhid1);
wkj_tmp = zeros(size(wkj));
wji = randn(nhid,ninpdim1);
olddelwkj=zeros(noutdim*nhid1); % weight of Wkj (J -> K)
olddelwji= zeros(ninpdim1*nhid);% weight of Wji (I -> J)

oi = zeros(ninpdim1,1);
oi(ninpdim1) = 1;   % output of data

sj = zeros(nhid,1);
sj(nhid) = 1;       % input of hidden layer j

oj = zeros(nhid1,1);
oj(nhid1) = 1;      % output of hidden layer j
sk = zeros(noutdim,1);        % input of output layer k
ok = zeros(noutdim,1);        % net output
dk = zeros(noutdim,1);        % desired output
 
Lowerlimit=0.02;
itermax=20000;
eta=0.7;            % (n -> eta -> learning rate)
beta=0.3;           % momentum term
 
iter=0;
error_avg=10;
 
title_text = sprintf('P3 IJK: %d X %d X %d \n iter = %d, eta = %f, beta = %f',ninpdim1,nhid1,noutdim,itermax,eta,beta);
file_text = sprintf('P3_IJK_%dX%dX%d_iter_%d_eta_%f',ninpdim1,nhid1,noutdim,itermax,eta,beta);


% internal variables
deltak = zeros(1,noutdim);
sumback = zeros(1,nhid);
deltaj = zeros(1,nhid);
 
while (error_avg > Lowerlimit) && (iter<itermax)
    iter=iter+1;
    error=0;
 
% Forward Computation:
            
    for ivector=1:nvectors
        oi=[data(ivector,1) data(ivector,2) 1]';
        dk=[data(ivector,3) data(ivector,4) data(ivector,5) data(ivector,6)]';
        for j=1:nhid
            sj(j)=wji(j,:)*oi;
            oj(j)=1/(1+exp(-sj(j)));    % sigmoid
        end
        oj(nhid1)=1.0;
 
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
 
         for j=1:nhid1
            for k=1:noutdim
               wkj_tmp(k,j)=wkj(k,j)+eta*deltak(k)*oj(j)+beta*olddelwkj(k,j);
               olddelwkj(k,j)=eta*deltak(k)*oj(j)+beta*olddelwkj(k,j);
            end
         end
 
         for j=1:nhid
            sumback(j)=0.0;
            for k=1:noutdim
               sumback(j)=sumback(j)+deltak(k)*wkj(k,j);
            end
            deltaj(j)=oj(j)*(1.0-oj(j))*sumback(j);
         end
 
 
         for i=1:ninpdim1
            for j=1:nhid
               wji(j,i)=wji(j,i)+eta*deltaj(j)*oi(i)+beta*olddelwji(j,i);
               olddelwji(j,i)=eta*deltaj(j)*oi(i)+beta*olddelwji(j,i);
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

plot(data(1:N,1),data(1:N,2),'ro');
plot(data(N+1:2*N,1),data(N+1:2*N,2),'bo');
plot(data(2*N+1:3*N,1),data(2*N+1:3*N,2),'go');
plot(data(3*N+1:4*N,1),data(3*N+1:4*N,2),'ko');

for ix=-5*4:1:20*4
    for iy=-5*4:1:20*4
        dx=0.25*(ix-1); 
        dy=0.25*(iy-1);
        oi=[dx dy 1]';
 
        for j=1:nhid
            sj(j)=wji(j,:)*oi;
            oj(j)=1/(1+exp(-sj(j)));
        end
        oj(nhid1)=1.0;
 
        for k=1:noutdim
            sk(k)=wkj(k,:)*oj;
            ok(k)=1/(1+exp(-sk(k)));
        end
 
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

