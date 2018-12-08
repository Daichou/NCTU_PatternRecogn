% Appendix K: Matlab Program of Multilayer Perceptron
 
% 15k_Appendix_K_mlpKY3_modify.m
% The program is written by Kou-Yuan Huang
 
clear all; 
% I=2+1;% J=3+1;% K=2;
nvectors=4;
ninpdim1=3;
nhid=3;
nhid1=4;
noutdim=2;
data(1,1)=0;data(1,2)=0; data(1,3)=1;data(1,4)=0; % 3,4 -> output
data(2,1)=1;data(2,2)=0; data(2,3)=0;data(2,4)=1;
data(3,1)=0;data(3,2)=1; data(3,3)=0;data(3,4)=1;
data(4,1)=1;data(4,2)=1; data(4,3)=1;data(4,4)=0;
 
%initialize
wkj = randn(2,4);
wkj_tmp = zeros(size(wkj));
wji = randn(3,3);
olddelwkj=[0 0 0 0; % weight of Wkj (J -> K)
           0 0 0 0];
olddelwji=[0 0 0;   % weight of Wji (I -> J)
           0 0 0;
           0 0 0];
oi = [0 0 1]';      % output of data
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
sumback = zeros(1,nhid);
deltaj = zeros(1,nhid);
 
while (error_avg > Lowerlimit) && (iter<itermax)
    iter=iter+1;
    error=0;
 
% Forward Computation:
            
    for ivector=1:nvectors
        oi=[data(ivector,1) data(ivector,2) 1]';
        dk=[data(ivector,3) data(ivector,4)]';
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
 
        % Real output
        if ok(1,1)<0.5
            plot(dx,dy, 'k .');
        elseif ok(1,1)>0.5
            plot(dx,dy, 'r .');
        end
    end
end

