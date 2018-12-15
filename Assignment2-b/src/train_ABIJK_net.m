
function [wkj,wji,wib,wba,error_r,ite] = train_ABIJK_net(data,eta,beta,layer,input,output,itermax,Lowerlimit)
    nvectors=length(data);
    ninpdim_with_bias=input+1;
    neuron_hid_layerJ=layer(1);
    neuron_hid_layerJ_with_bias=neuron_hid_layerJ+1;
    neuron_hid_layerI=layer(2);
    neuron_hid_layerI_with_bias=neuron_hid_layerI+1;
    neuron_hid_layerB=layer(3);
    neuron_hid_layerB_with_bias=neuron_hid_layerB+1;
    noutdim=output;

    %initialize
    wkj = normrnd(0,sqrt(2/(input+output)),noutdim,neuron_hid_layerI_with_bias);
    wji = normrnd(0,sqrt(2/(input+output)),neuron_hid_layerJ_with_bias,neuron_hid_layerI_with_bias);
    wib = normrnd(0,sqrt(2/(input+output)),neuron_hid_layerI_with_bias,neuron_hid_layerB_with_bias);
    wba = normrnd(0,sqrt(2/(input+output)),neuron_hid_layerB_with_bias,ninpdim_with_bias);
    wkj_tmp = zeros(size(wkj));
    wji_tmp = zeros(size(wji));
    wib_tmp = zeros(size(wib));
    wba_tmp = zeros(size(wba));
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

    iter=0;
    error_avg=10;

    % internal variables
    deltak = zeros(1,noutdim);
    deltaj = zeros(1,neuron_hid_layerJ_with_bias);
    deltai = zeros(1,neuron_hid_layerI_with_bias);
    deltab = zeros(1,neuron_hid_layerB_with_bias);
    sumback = zeros(1,max(neuron_hid_layerJ_with_bias, max(neuron_hid_layerI_with_bias,neuron_hid_layerB_with_bias)));
    counter = 0;
    while (error_avg > Lowerlimit) && (iter<itermax)
        iter=iter+1;
        error=0;
        data_index = randperm(length(data));
        if counter > 1000
            counter = 0;
            eta = eta*0.9;
            beta = beta*0.9;
        end
        counter = counter + 1;
    % Forward Computation:
        for ivector=1:nvectors
            rvector = data_index(ivector);
            oa=single([data(rvector,1:input) 1]');
            dk=single([data(rvector,input+1:output+input)]');

            for j=1:neuron_hid_layerB
                sb(j)=wba(j,:)*oa;
                ob(j)=Activation(sb(j));    % sigmoid
            end
            ob(neuron_hid_layerB_with_bias)=1.0;

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
            
            %error=error+ (dk-ok)' *(dk-ok)/2;
            error=error+sum(abs(dk-ok)); % abs is absolute each element
            
    % Backward learning:
     
             for k=1:noutdim
                deltak(k)=(dk(k)-ok(k))*deActivation(ok(k)); % gradient term
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
     
             for j=1:neuron_hid_layerB_with_bias
                for k=1:neuron_hid_layerI
                   wbi(k,j)=wib(k,j)+eta*deltai(k)*ob(j)+beta*olddelwib(k,j);
                   olddelwib(k,j)=eta*deltai(k)*ob(j)+beta*olddelwib(k,j);
                end
             end

             for j=1:neuron_hid_layerB
                sumback(j)=0.0;
                for k=1:neuron_hid_layerI_with_bias
                   sumback(j)=sumback(j)+deltai(k)*wib(k,j);
                end
                deltab(j)=deActivation(ob(j))*sumback(j);
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
end

function o = Activation(s)
    o = ReLu(s);
end

function o = deActivation(s)
    o = deReLu(s);
end
