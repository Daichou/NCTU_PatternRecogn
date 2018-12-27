clear all;

f = [0.1 0.1 0.1 0.1;0.1 0.1 0.1 0.1;0.1 0.1 0.1 0.1;0.1 0.1 0.1 0.1;];
g = [0.1 0.1 0.1 0.1;0.1 0.1 0.1 0.1;0.1 0.1 0.1 0.1;0.1 0.1 0.1 0.1;];

w = 4;
h = 4;

c = zeros(4,4);

for j=1:1:(w+h-1)
    for k=1:1:(w+h-1)
        sum = 0;

        for p=max(1,j+1-w):1:min(j,w)
            for q=max(1,k+1-h):1:min(k,h)
                sum = sum + f(p,q)*g(j-p+1,k-q+1);
            end
        end

        c(j,k) = sum;
    end
end

c
