
%clear all
addpath(genpath('../'))

n = 30;
m = 500;


% generate process information
xTrue = rand(n,1);
tQ = randn(n);
Q = eye(n) + tQ'*tQ;%Q=Q*10000;
L = chol(Q)';

% generate measurement information
sigma = 2;
H = randn(m, n);


% generate measurements
z = H*L*xTrue + sigma*randn(m,1);

fun = @(x) linearMat(x, H);

% constraint = 'box';
% boxSize = 1e10;%0.01;
% % Define constraints: say our parameters are in a big box
% switch(constraint)
%     case 'box'
%         A = [speye(n); -speye(n)];
%         params.a = boxSize*ones(2*n, 1);
%         params.A = A';
%     case 'pos'
%         A = -speye(n);
%         params.a = zeros(n, 1);
%         params.A = A';
%     case 'none'
%         A = [];
%         a = [];
%     otherwise error('Unknown constraint');
% end



params.pFlag = 1;
params.constraints = 0;

params.m = m;
params.n = n;
params.proc_mMult = 0.5;
params.meas_mMult = 0.5;

params.proc_lambda = 1;
params.meas_lambda = 1;

params.meas_eps = 0.2;

par.kappa = 0.01;
par.mMult = 1/par.kappa;
%par.lambda = 1/par.kappa;
%plotPenalty('huber', par);


%% Vapnik-L2 and test
%fs = sysIDFunc(z,Q,H,sigma, 'vapnik', 'l2', params);

fs = sysIDFunc(z,Q,fun,sigma, 'vapnik', 'l2', params);

cvx_begin 
  variables f(n)
  minimize( sum(pos(z - H*f - params.meas_eps)) + sum(pos(-z + H*f - params.meas_eps)) + sigma^2*f'* inv(Q) * f)  
cvx_end 



%% L2-L2 and test
% fs = sysIDFunc(z,Q,H,sigma, 'l2', 'l2', params);
% 
% cvx_begin 
%   variables f(n)
%   minimize( sum_square(z - H*f) + sigma^2*f'* inv(Q) * f)  
% cvx_end 

%% L1-L1 and test
%fs = sysIDFunc(z,Q,H,sigma, 'l1', 'l2', params);

% cvx_begin 
%   variables f(n)
%   minimize( norm(z - H*f, 1) + sigma^2*f'* inv(Q) * f)  
% cvx_end 


 

%% prints test results. 
valCVX = norm(z - H*f, 1) + sigma^2*f'*inv(Q)*f;
valIP = norm(z - H*fs, 1) + sigma^2*fs'*inv(Q)*fs;

fprintf('CVX value: %5.3f, IP value: %5.3f, inf norm diff: %5.3f\n', valCVX, valIP, norm(f - fs, inf));




