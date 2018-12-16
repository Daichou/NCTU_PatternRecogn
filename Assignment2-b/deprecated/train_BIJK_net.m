function [wkj,wji,wib,error_r,ite] = train_BIJK_net(data,eta,beta,layer,input,output,itermax,Lowerlimit)
    nvectors=length(data);
    ninpdim_with_bias=input+1;
    neuron_hid_layerJ=layer(1);
    neuron_hid_layerJ_with_bias=neuron_hid_layerJ+1;
    neuron_hid_layerI=layer(2);
    neuron_hid_layerI_with_bias=neuron_hid_layerI+1;
    noutdim=output;

    %initialize
    wkj = normrnd(0,sqrt(2/(input+output)),noutdim,neuron_hid_layerJ_with_bias);
    wji = normrnd(0,sqrt(2/(input+output)),neuron_hid_layerI_with_bias,neuron_hid_layerJ_with_bias);
    wib = normrnd(0,sqrt(2/(input+output)),neuron_hid_layerI_with_bias,ninpdim_with_bias);
    wkj_tmp = zeros(size(wkj));
    wji_tmp = zeros(size(wji));
    wib_tmp = zeros(size(wib));
    olddelwkj=zeros(noutdim , neuron_hid_layerJ_with_bias); % weight of Wkj (J -> K)
    olddelwji=zeros(neuron_hid_layerI_with_bias , neuron_hid_layerJ_with_bias);   % weight of Wji (I -> J)
    olddelwib=zeros(neuron_hid_layerI_with_bias,ninpdim_with_bias);   % weight of Wji (B -> I)
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
    sumback = zeros(1,max(neuron_hid_layerJ_with_bias, neuron_hid_layerI_with_bias));

    while (error_avg > Lowerlimit) && (iter<itermax)
        iter=iter+1;
        error=0;
    % Forward Computation:
        r_index = randperm(length(data));
        for ivector=1:nvectors
            rvector = r_index(ivector);
            ob=single([data(rvector,1:input) 1]');
            dk=single[data(rvector,input+1:input+output)]');

            for j=1:neuron_hid_layerI
                si(j)=wib(j,:)*ob;
                oi(j)=Activation(si(j));    % sigmoid
            end
            oi(neuron_hid_layerI_with_bias)=1.0;

            for j=1:neuron_hid_layerJ
                sj(j)=wji(j,:)*oi;
                oj(j)=Activation(sj(j));    % sigmoid
            end
            oj(neuron_hid_layerJ_with_bias)=1.0;
 
            for k=1:noutdim
                sk(k)=wkj(k,:)*oj;
                ok(k)=Activation(sk(k));    % signmoid
            end
           
            error=error+sum(abs(dk-ok)); % abs is absolute each element
 
    % Backward learning:

            for k=1:noutdim
               deltak(k)=(dk(k)-ok(k))*deActivation(ok(k)); % gradient term
            end

            for j=1:neuron_hid_layerJ_with_bias
               for k=1:noutdim
                  wkj(k,j)=wkj(k,j)+eta*deltak(k)*oj(j)+beta*olddelwkj(k,j);
                  olddelwkj(k,j)=eta*deltak(k)*oj(j)+beta*olddelwkj(k,j);
               end
            end
     
            for j=1:neuron_hid_layerJ
               sumback(j)=0.0;
               for k=1:noutdim
                  sumback(j)=sumback(j)+deltak(k)*wkj(k,j);
               end
               deltaj(j)=deActivation(oj(j))*sumback(j);
            end

            for j=1:neuron_hid_layerI_with_bias
               for k=1:neuron_hid_layerJ
                  wji(k,j)=wji(k,j)+eta*deltaj(k)*oi(j)+beta*olddelwji(k,j);
                  olddelwji(k,j)=eta*deltaj(k)*oi(j)+beta*olddelwji(k,j);
               end
            end

            for j=1:neuron_hid_layerI
               sumback(j)=0.0;
               for k=1:neuron_hid_layerJ_with_bias
                  sumback(j)=sumback(j)+deltaj(k)*wji(k,j);
               end
               deltai(j)=deActivation(oi(j))*sumback(j);
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

function o = Activation(s)
    o = ReLu(s);
end

function o = deActivation(s)
    o = deReLu(s);
end
