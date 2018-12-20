function ok = FeedFoward_IJK(wji,wkj,oi,input,output,layer,method)
    ninpdim_with_bias=input+1;
    neuron_hid_layerJ=layer(1);
    neuron_hid_layerJ_with_bias=neuron_hid_layerJ+1;
    noutdim=output;

    sj = zeros(neuron_hid_layerJ_with_bias,1);      % input of hidden layer j
    oj = zeros(neuron_hid_layerJ_with_bias,1);
    oj(neuron_hid_layerJ_with_bias) = 1;    % output of hidden layer j

    sk = zeros(neuron_hid_layerJ_with_bias,1);        % input of output layer k
    ok = zeros(noutdim,1);        % net output
    dk = zeros(noutdim,1);        % desired output

    for j=1:neuron_hid_layerJ
        sj(j)=wji(j,:)*oi;
        oj(j)=Activation(sj(j),method);    % sigmoid
    end
    oj(neuron_hid_layerJ_with_bias)=1.0;

    for k=1:noutdim
        sk(k)=wkj(k,:)*oj;
        ok(k)=Activation(sk(k),method);    % signmoid
    end
end

function o = Activation(s,method)
    if method == 1
        o = Sigmoid(s);
    else
        o = ReLu(s);
    end
end
