
x=0:0.01:pi;
for i = 1:5
    y_plot=sin(x*pi*i);
    plot(x,y_plot,'.k');
    axis([0,pi,-1.2,1.2]);
    title(strcat('P17 T/',num2str(i)))
    pause(5);
end
