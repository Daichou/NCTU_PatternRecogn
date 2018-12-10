function [wkj,wji,wib,error_r,ite] = train_BIJK_net(data,eta,beta,layer,input,output,itermax,Lowerlimit)
    nvectors=length(data);
    ninpdim_with_bias=input+1;
    neuron_hid_layerJ=layer(1);
    neuron_hid_layerJ_with_bias=neuron_hid_layerJ+1;
    neuron_hid_layerI=layer(2);
    neuron_hid_layerI_with_bias=neuron_hid_layerI+1;
    noutdim=output;

    %initialize
    wkj = randn(noutdim,neuron_hid_layerI_with_bias);
    wji = randn(neuron_hid_layerJ_with_bias,neuron_hid_layerI_with_bias);
    wib = randn(neuron_hid_layerI_with_bias,neuron_hid_layerB_with_bias);
    wkj_tmp = zeros(size(wkj));
    wji_tmp = zeros(size(wji));
    wib_tmp = zeros(size(wib));
    olddelwkj=zeros(noutdim , neuron_hid_layerJ_with_bias); % weight of Wkj (J -> K)
    olddelwji=zeros(neuron_hid_layerJ_with_bias , neuron_hid_layerI_with_bias);   % weight of Wji (I -> J)
    olddelwib=zeros(neuron_hid_layerB_with_bias , neuron_hid_layerI_with_bias);   % weight of Wji (B -> I)
    ob = zeros(ninpdim_with_bias,1);
    ob(ninpdim_with_bias) = 1;

    si = zeros(ninpdim_with_bias,1);       % input of hidden layer i
    oi = zeros(neuron_hid_layerJ_with_bias,1);
    oi(neuron_hid_layerI_with_bias) = 1;    % output of hidden layer i

    sj = zeros(neuron_hid_layerI_with_bias,1);      % input of hidden layer j
    oj = zeros(neuron_hid_layerJ_with_bias,1);
    oj(neuron_hid_layerJ_with_bias) = 1;    % output of hidden layer j

    sk = zeros(neuron_hid_layerJ_with_bias,1);        % input of output layer k
    ok = zeros(noutdim,1);        % net output
    dk = zeros(noutdim,1);        % desired output

    iter=0;
    error_avg=10;

    % internal variables
    deltak = zeros(1,noutdim);
    deltaj = zeros(1,neuron_hid_layerJ_with_bias);
    deltai = zeros(1,neuron_hid_layerI_with_bias);
    sumback = zeros(1,max(neuron_hid_layerJ_with_bias, max(neuron_hid_layerI_with_bias)));

    while (error_avg > Lowerlimit) && (iter<itermax)
        iter=iter+1;
        error=0;
     
    % Forward Computation:
        for ivector=1:nvectors
            ob=[data(ivector,1) data(ivector,2) 1]';
            dk=[data(ivector,3) data(ivector,4)]';

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

            for j=1:neuron_hid_layerI_with_bias
               for k=1:neuron_hid_layerJ
                  wji_tmp(k,j)=wji(k,j)+eta*deltaj(k)*oi(j)+beta*olddelwji(k,j);
                  olddelwji(k,j)=eta*deltaj(k)*oi(j)+beta*olddelwji(k,j);
               end
            end

            for j=1:neuron_hid_layerI
               sumback(j)=0.0;
               for k=1:neuron_hid_layerJ_with_bias
                  sumback(j)=sumback(j)+deltaj(k)*wji(k,j);
               end
               deltai(j)=oi(j)*(1.0-oi(j))*sumback(j);
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
end
