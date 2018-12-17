function [wkj,wji,error_r,ite,time_r] = train_IJK_net(data,eta,beta,layer,input,output,itermax,Lowerlimit,method)
    nvectors=length(data);
    ninpdim_with_bias=input+1;
    neuron_hid_layerJ=layer(1);
    neuron_hid_layerJ_with_bias=neuron_hid_layerJ+1;
    noutdim=output;

    %initialize
    wkj = normrnd(0,sqrt(2/(input+output)),noutdim,neuron_hid_layerJ_with_bias);
    wji = normrnd(0,sqrt(2/(input+output)),neuron_hid_layerJ_with_bias,ninpdim_with_bias);
    wkj_tmp = zeros(size(wkj));
    wji_tmp = zeros(size(wji));
    olddelwkj=zeros(noutdim , neuron_hid_layerJ_with_bias); % weight of Wkj (J -> K)
    olddelwji=zeros(neuron_hid_layerJ_with_bias , ninpdim_with_bias);   % weight of Wji (I -> J)
    oi = zeros(ninpdim_with_bias,1);
    oi(ninpdim_with_bias) = 1;

    sj = zeros(neuron_hid_layerJ_with_bias,1);      % input of hidden layer j
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
    sumback = zeros(1,neuron_hid_layerJ_with_bias);

    init_time = cputime;

    while (error_avg > Lowerlimit) && (iter<itermax)
        iter=iter+1;
        error=0;
    % Forward Computation:
        data_index = randperm(length(data));
        for ivector=1:nvectors
            rvector = data_index(ivector);
            oi=single([data(rvector,1:input) 1]');
            dk=single([data(rvector,input+1:input+output)]');

            for j=1:neuron_hid_layerJ
                sj(j)=wji(j,:)*oi;
                oj(j)=Activation(sj(j),method);    % sigmoid
            end
            oj(neuron_hid_layerJ_with_bias)=1.0;
 
            for k=1:noutdim
                sk(k)=wkj(k,:)*oj;
                ok(k)=Activation(sk(k),method);    % signmoid
            end

            %error=error+ (dk-ok)' *(dk-ok)/2;
            error=error+sum(abs(dk-ok)); % abs is absolute each element
 
    % Backward learning:

            for k=1:noutdim
               deltak(k)=(dk(k)-ok(k))*deActivation(ok(k),method); % gradient term
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
               deltaj(j)=deActivation(oj(j),method)*sumback(j);
            end


            for i=1:ninpdim_with_bias
                for j=1:neuron_hid_layerJ
                   wji(j,i)=wji(j,i)+eta*deltaj(j)*oi(i)+beta*olddelwji(j,i);
                   olddelwji(j,i)=eta*deltaj(j)*oi(i)+beta*olddelwji(j,i);
                end
             end
             wkj = wkj_tmp;

        end
        
        ite(iter)=iter;
        error_avg=error/nvectors;
        error_r(iter)=error_avg;
        time_r(iter)=cputime-init_time;
    end
end

function o = Activation(s,method)
    if method == 1
        o = Sigmoid(s);
    else
        o = ReLu(s);
    end
end

function o = deActivation(s,method)
    if method == 1
        o = deSigmoid(s);
    else
        o = deReLu(s);
    end
end
