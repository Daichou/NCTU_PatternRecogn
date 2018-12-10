
function ok = FeedFoward(wba,wib,wji,wkj,oa,input,output,layer)
    ninpdim_with_bias=input+1;
    neuron_hid_layerJ=layer(1);
    neuron_hid_layerJ_with_bias=neuron_hid_layerJ+1;
    neuron_hid_layerI=layer(2);
    neuron_hid_layerI_with_bias=neuron_hid_layerI+1;
    neuron_hid_layerB=layer(3);
    neuron_hid_layerB_with_bias=neuron_hid_layerB+1;
    noutdim=output;
    sb = zeros(neuron_hid_layerB_with_bias,1);
    ob = zeros(neuron_hid_layerB_with_bias,1);
    ob(neuron_hid_layerB_with_bias) = 1;

    si = zeros(ninpdim_with_bias,1);       % input of hidden layer i
    oi = zeros(neuron_hid_layerJ_with_bias,1);
    oi(neuron_hid_layerI_with_bias) = 1;    % output of hidden layer i

    sj = zeros(neuron_hid_layerI_with_bias,1);      % input of hidden layer j
    oj = zeros(neuron_hid_layerJ_with_bias,1);
    oj(neuron_hid_layerJ_with_bias) = 1;    % output of hidden layer j

    sk = zeros(neuron_hid_layerJ_with_bias,1);        % input of output layer k
    ok = zeros(noutdim,1);        % net output

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
end
