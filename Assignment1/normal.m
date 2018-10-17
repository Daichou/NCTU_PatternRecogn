s1 = uint64(datenum(datetime)+cputime*1000);
s2 = uint64(datenum(datetime)+cputime*1000);
mean = 0;
std = 1;
normals = []
for j = 1:100000
    c = [];
    for i = 1:2
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
    n_final = sqrt(-2*log(c(1)))*cos(2*pi*c(2))* std + mean;
    normals = cat(1,normals,n_final);
end
histogram(normals,100);
h = chi2gof(normals)