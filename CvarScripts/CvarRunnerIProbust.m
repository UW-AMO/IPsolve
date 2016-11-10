cd ..;
clear all;
addpath(genpath(pwd))
cd Scripts;
rng(1,'twister'); % set random seed and generator
% n = 20;  % number of stocks
% m = 100; % time measurements per stock
% Y = randn(m, n);  %gaussian samples. 

load('DJIAData');
Y=-DJIAReturns;
m=size(Y,1);
n=size(Y,2);


goodcolumn=1;
badcell=10;
Y(:,goodcolumn)=Y(:,goodcolumn)-0.2;
Y(badcell,goodcolumn)=Y(badcell,goodcolumn)+20;




%%
E = ones(n,1); % the E from the writeup
b = zeros(m,1);
% form augmented system 
Yaug = [Y -ones(m,1)]; % for - alpha in objective
linTerm = [zeros(n,1); 1]; % pick out alpha coordinate
q = m; % this is the number of datapoints
bet = 0.9; % the quantile

%% robust control
kEx = 10; % days to exclude;
params.trim = 1;
params.h = m - kEx;
alpha = 1-kEx/m;
% robust weight
params.meas_lambda = 1/(q*(alpha-bet)); % this is the hinge loss tilt

% more information for IPsolve. 
params.procLinear = 0;
params.silent = 0;
params.inexact = 0;
params.optTol = 1e-8;
params.progTol = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% constarints interface to IPsolve
% params.constraints = 1; 
% A = [-speye(n); ones(1,n); -ones(1,n)]; % contraint matrix
% A = [A zeros(n+2,1)]; 
% a = [zeros(n,1); 1; -1]; % cosntraint vector 
% params.A = A'; 
% params.a = a;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

params.constraints = 1; 

A = [-speye(n); ones(1,n); -ones(1,n)]; % constraints (matrix)
A = [A zeros(n+2,1)];
a = [zeros(n,1); 1; -1]; %  (constraints, vector)
    
params.A = A'; 
params.a = a;



params.rho = 0;
params.delta = 0;

params.inexact = 0;
[xIP_CVaR] = run_example(Yaug, b, 'hinge', [], linTerm, params);


kEx = 10; % days to exclude;
alpha = 1-kEx/m; 
% robust weight
params.meas_lambda = 1/(q*(alpha-bet)); % this is the hinge loss tilt


[xIP_RCVaR] = run_example(Yaug, b, 'hinge', [], linTerm, params);

[x_CVaR, x_RCVaR]= RobustCVaR(Y, bet, alpha);

fprintf('IP CVAR, CVX CVAR, IP Robust CVAR, CVX Robust CVAR\n') 
% [xIP_CVaR(1:n)/xIP_CVaR(1) x_CVaR/x_CVaR(1) xIP_RCVaR(1:n)/xIP_RCVaR(1) x_RCVaR/x_RCVaR(1)]
[xIP_CVaR(1:n) x_CVaR xIP_RCVaR(1:n) x_RCVaR]

%[xIP] = run_example(A, b, 'l2', 'l1', params);


%% Check Results
ret_RCVaR = Y*x_RCVaR;
outlier_idx=find(ret_RCVaR>=quantile(ret_RCVaR,alpha));
outlier_Date=Date{outlier_idx}
Y(outlier_idx,:)

rowmeans=mean(Y,2);
rowmeans_outlier_idx=find(rowmeans>=quantile(rowmeans,alpha));
fprintf('IP CVAR, CVX CVAR, IP Robust CVAR, CVX Robust CVAR\n') 

ret_IP_RCVaR = Y*xIP_RCVaR(1:n);
outlier_IP_idx=find(ret_IP_RCVaR>=quantile(ret_IP_RCVaR,alpha));
outlier_IP_Date=Date{outlier_IP_idx}
Y(outlier_IP_idx)

fprintf('IP_RCVaR CVX_RCVaR NaiveMean\n')
[outlier_IP_idx outlier_idx rowmeans_outlier_idx]

% now the outlier picked out by the robust CVaR optimization is different
% from the row with the maximum mean value, which shows how effective our
% approach is.

