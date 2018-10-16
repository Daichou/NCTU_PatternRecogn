s1 = uint64(datenum(datetime)+cputime*1000);
s2 = uint64(datenum(datetime)+cputime*1000);

for i = 1:5
x = uint64(s1);
y = uint64(s2);
s1 = y;
x = bitxor(x,bitshift(x,32,'uint64'),'uint64');

s2 = bitxor(bitxor(bitxor(x,y,'uint64'),bitshift(x,-17,'uint64'),'uint64'),bitshift(y,-26,'uint64'),'uint64');
s2
y
ans = double(s2 + y)
bot = double(2^64-1)
final = double(ans/bot)
end