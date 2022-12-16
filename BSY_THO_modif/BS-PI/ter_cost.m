function out = ter_cost(x)
x(1) = 0;
load('data2')
%xs = [1;2.95000000000000;0.950000000000000;380;375;20;-5000];
%x = x./xs;
%xxx = xxx./xs;
dis = 1./sum((x - xxx).*(x - xxx));
dis = dis.^2;
%dis = 1./sum((x - xxx).*(x - xxx));
%z = sum((x - xxx).*(x - xxx)); % 1st norm 
out = (dis*ppp'/(sum(dis)));