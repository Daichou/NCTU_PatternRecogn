clear all; 
% B = 2+1; % I=3+1;% J=3+1;% K=2;
nvectors=4;
ninpdim1=3;
neuron_hid_layerJ_with_bias=3;
neuron_hid_layerJ_with_bias1=4;
neuron_hid_layerI_with_bias=3;
neuron_hid_layerI_with_bias1=4;
noutdim=2;
data(1,1)=0;data(1,2)=0; data(1,3)=1;data(1,4)=0; % 3,4 -> output
data(2,1)=1;data(2,2)=0; data(2,3)=0;data(2,4)=1;
data(3,1)=0;data(3,2)=1; data(3,3)=0;data(3,4)=1;
data(4,1)=1;data(4,2)=1; data(4,3)=1;data(4,4)=0;

%initialize
wkj = randn(2,4);
wkj_tmp = zeros(size(wkj));
wji_tmp = zeros(size(wkj));
wji = randn(4,4);
wib = randn(3,3);
olddelwkj=zeros(2*4); % weight of Wkj (J -> K)
olddelwji=zeros(3*3);   % weight of Wji (I -> J)
olddelwib=zeros(3*3);   % weight of Wji (I -> J)
ob = [0 0 1];       % output of data
si = [0 0 0];       % input of hidden layer i
oi = [0 0 0 1]';    % output of hidden layer i
sj = [0 0 0]';      % input of hidden layer j
oj = [0 0 0 1]';    % output of hidden layer j
sk = [0 0]';        % input of output layer k
ok = [0 0]';        % net output
dk = [0 0]';        % desired output

Lowerlimit=0.02;
itermax=20000;
eta=0.7;            % (n -> eta -> learning rate)
beta=0.3;           % momentum term
 
iter=0;
error_avg=10;
 
 
% internal variables
deltak = zeros(1,noutdim);
deltaj = zeros(1,neuron_hid_layerJ_with_bias);
deltai = zeros(1,neuron_hid_layerI_with_bias);
sumback = zeros(1,max(neuron_hid_layerJ_with_bias,neuron_hid_layerI_with_bias));
 
while (error_avg > Lowerlimit) && (iter<itermax)
    iter=iter+1;
    error=0;
 
% Forward Computation:
            
    for ivector=1:nvectors
        ob=[data(ivector,1) data(ivector,2) 1]';
        dk=[data(ivector,3) data(ivector,4)]';

        for j=1:neuron_hid_layerI_with_bias
            si(j)=wib(j,:)*ob;
            oi(j)=1/(1+exp(-si(j)));    % sigmoid
        end
        oi(neuron_hid_layerJ_with_bias1)=1.0;

        for j=1:neuron_hid_layerJ_with_bias
            sj(j)=wji(j,:)*oi;
            oj(j)=1/(1+exp(-sj(j)));    % sigmoid
        end
        oj(neuron_hid_layerJ_with_bias1)=1.0;
 
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
 
         for j=1:neuron_hid_layerJ_with_bias1
            for k=1:noutdim
               wkj_tmp(k,j)=wkj(k,j)+eta*deltak(k)*oj(j)+beta*olddelwkj(k,j);
               olddelwkj(k,j)=eta*deltak(k)*oj(j)+beta*olddelwkj(k,j);
            end
         end
 
         for j=1:neuron_hid_layerJ_with_bias
            sumback(j)=0.0;
            for k=1:noutdim
               sumback(j)=sumback(j)+deltak(k)*wkj(k,j);
            end
            deltaj(j)=oj(j)*(1.0-oj(j))*sumback(j);
         end

         for j=1:neuron_hid_layerI_with_bias
            sumback(j)=0.0;
            for k=1:neuron_hid_layerJ_with_bias
               sumback(j)=sumback(j)+deltaj(k)*wji(k,j);
            end
            deltai(j)=oi(j)*(1.0-oi(j))*sumback(j);
         end
 
         for i=1:ninpdim1
            for j=1:neuron_hid_layerJ_with_bias
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
 
 
figure;
hold on;
plot(ite, error_r);
 
figure;
hold on;
for n=1:3:4
    plot(data(n,1), data(n,2),'r o');
end
for n=2:1:3
    plot(data(n,1), data(n,2),'k s');
end
 
for ix=1:1:51
    for iy=1:1:51
        dx=0.02*(ix-1); 
        dy=0.02*(iy-1);
        ob=[dx dy 1]';
 
        for j=1:neuron_hid_layerI_with_bias
            si(j)=wib(j,:)*ob;
            oi(j)=1/(1+exp(-si(j)));    % sigmoid
        end
        oi(neuron_hid_layerJ_with_bias1)=1.0;

        for j=1:neuron_hid_layerJ_with_bias
            sj(j)=wji(j,:)*oi;
            oj(j)=1/(1+exp(-sj(j)));    % sigmoid
        end
        oj(neuron_hid_layerJ_with_bias1)=1.0;
 
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

