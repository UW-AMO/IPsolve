% Written by Aleksandr Aravkin (saravkin at us dot ibm dot com)
% L1-regularized Huber example. 
clear all;
%Generate problem data
randn('seed', 0);
rand('seed',0);


% choose which ADMM program to run; whether to 
% to use smooth version (1, runs huberl1smooth) 
% or nonsmooth (2, runs huberl1)
smooth = 0;

% Choose whether the problem is well conditioned (1) or ill-conditioned (0)
wellCond = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = 200;       % number of examples
n = 50;       % number of features

p = 1;      % sparsity density

x0 = randn(n, 1);
A = randn(m,n);

b = A*x0 + sqrt(0.001)*randn(m,1);
%b = b + 10*sprand(m,1,300/m);      % add sparse, large noise

params.procLinear = 0;

%Solve problem


params.meas_kappa = 1;
%params.uConstraints = 1;
params.meas_scale = 2;
% params.uMax = params.meas_scale;
% params.uMin = -params.meas_scale;


boxSize = 2;
params.constraints = 1; 
Con = [speye(n); -speye(n)];
params.a = boxSize*ones(2*n, 1);
params.A = Con';


[xIP] = run_example(A, b, 'hybrid', [], params);
%Reporting

cvxFun = @(x)hybridFuncCVX(x, params.meas_scale);
tic
 cvx_begin
   variables xCVX(n)
   minimize( cvxFun(b - A*xCVX))  
   subject to 
   norm(xCVX, inf) <= boxSize
 cvx_end
toc


fprintf('Our objective: %5.3f, CVX objective: %5.3f\n', cvxFun(xCVX), cvxFun(xIP));


 fprintf('Inf norm between our solution and CVX solution: %5.3f\n', norm(xCVX - xIP, inf));
%[xIP xCVX]
