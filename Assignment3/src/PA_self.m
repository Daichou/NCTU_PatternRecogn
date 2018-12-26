clear all;
f = [0.1 0.1 0.1 0.1];
g = [0.1 0.1 0.1 0.1];

m = length(f);
n = length(g);

for k=1:1:(m+n-1)
    delta_t = k*0.1;
    
    sum_w = 0;
    for j = max(1,k+1-n):1:min(k,m)
        sum_w = sum_w + f(j)*g(k-j+1);
    end
    w(k) = sum_w;
end
w