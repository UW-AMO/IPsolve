

clear g h
Po=eye(2);%[10 0;0 10];%=Vo
mu=[0;0];%=xo=x1
N = 1000;
for i=1:N
    g{i}=[0.6 0;2 0];
    %Q{i}=[1 0;0 1];
    Q{i} = [0 0; 0 1];
    h{i}=[0.8 0];
    R{i}=1;
    y{i}=randn;
end
[Xs,Ps]=kalman(mu,Po,y,g,Q,h,R);

figure()
plot(Xs(1,:))