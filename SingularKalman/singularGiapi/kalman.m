
%Xs e Ps: smoothed estimates and convariances

%%%%Kalman filter
%mu=x(1|0)
%Po=P(1|0)
%[mu,Po,y{1},R{1}]......(g{1},Q{1}).....[x2,y{2},R{2}].......(g{2},Q{2})...



function [Xs,Ps]=kalman(mu,Po,y,g,Q,h,R)

warning off

n=length(y);%number of time instants where data are collected
nx=size(Po,1);%state space dimension


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% FORWARD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% TIME UPDATE
Xp(:,1)=mu;%stima stato predetta
Pp{1}=Po;%covarianza stato predetta
for i=1:n
    %%%%%%%%%%%%%%% MEASUREMENT UPDATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Ve{i}=h{i}*Pp{i}*h{i}'+R{i};%innovation variance
    e{i}=y{i}-h{i}*Xp(:,i);%innovation
    Xf(:,i)=Xp(:,i)+Pp{i}*h{i}'*pinv(Ve{i})*e{i};
    Pf{i}=Pp{i}-Pp{i}*h{i}'*pinv(Ve{i})*h{i}*Pp{i};
    %%%%%%%%%%%%%%%%%%% TIME UPDATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if i<n
        Xp(:,i+1)=g{i}*Xf(:,i);
        Pp{i+1}=g{i}*Pf{i}*g{i}'+Q{i};
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% BACKWARD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% TIME UPDATE
Xs(:,n)=Xf(:,n);
Ps{n}=Pf{n};
for j=1:(n-1)
    i=n-j;
    A{i}=Pf{i}*g{i}'*pinv(Pp{i+1});
    Xs(:,i)=Xf(:,i)+A{i}*(Xs(:,i+1)-Xp(:,i+1));
    Ps{i}=Pf{i}+A{i}*(Ps{i+1}-Pp{i+1})*A{i}';
end
