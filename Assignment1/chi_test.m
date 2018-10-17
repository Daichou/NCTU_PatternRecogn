N=100; % sample size
a=0; % lower boundary
b=1; % higher boundary
x=unifrnd(a,b,N,1);
%x(x<.9) = rand(sum(x<.9),1);
nbins = 10; % number of bin
edges = linspace(a,b,nbins+1); % edges of the bins
E = N/nbins*ones(nbins,1); % expected value (equal for uniform dist)

[h,p,stats] = chi2gof(x,'Expected',E,'Edges',edges)