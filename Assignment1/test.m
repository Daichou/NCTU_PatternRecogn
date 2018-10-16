
x=0:0.05:4*pi;
for i = 1:5
    y_plot=sin(x*pi/i);
    plot(x,y_plot,'.k');
    axis([0,4*pi,-1.2,1.2]);
    title(strcat('P17 T/',num2str(i)))
    pause(5);
end
