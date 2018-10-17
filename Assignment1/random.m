s1 = uint64(datenum(datetime)+cputime*1000)
s2 = uint64(datenum(datetime)+cputime*1000)
c = []
for i = 1:100
    x = uint64(s1);
    y = uint64(s2);
    s1 = y;
    x = bitxor(x,bitshift(x,32,'uint64'),'uint64');

    s2 = bitxor(bitxor(bitxor(x,y,'uint64'),bitshift(x,-17,'uint64'),'uint64'),bitshift(y,-26,'uint64'),'uint64');
    ans = double(s2) + double(y);
    if (ans > 2^64-1)
       ans = ans - 2^64+1;
    end
    bot = double(2^64-1);
    final = double(ans/bot);
    c = cat(1,c,final);
end
histogram(c,100);
a=0; % lower boundary
b=1; % higher boundary
%x=unifrnd(a,b,N,1);
%x(x<.9) = rand(sum(x<.9),1);
nbins = 10; % number of bin
edges = linspace(a,b,nbins+1); % edges of the bins
E = N/nbins*ones(nbins,1); % expected value (equal for uniform dist)

[h,p,stats] = chi2gof(c,'Expected',E,'Edges',edges)
