% Written by Aleksandr Aravkin (saravkin at us dot ibm dot com)
% L1-regularized Huber example. 
%clear all;
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

m = 100;       % number of examples
n = 50;       % number of features

p = 1;      % sparsity density

x0 = randn(n, 1);
A = randn(m,n);

b = A*x0 + sqrt(0.001)*randn(m,1);
%b = b + 10*sprand(m,1,300/m);      % add sparse, large noise

params.procLinear = 0;
params.kappa = 1;
lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;

%Solve problem


params.proc_lambda = lambda;
params.meas_kappa = 1;
params.uConstraints = 1;
params.meas_scale = 1;
params.uMax = params.meas_scale;
params.uMin = -params.meas_scale;

[xIP] = run_example(A, b, 'hybrid', [], params);
%Reporting

cvxFun = @(x)hybridFuncCVX(x, params.meas_scale);

 cvx_begin
   variables xCVX(n)
   minimize( cvxFun(b - A*xCVX))  
 cvx_end

 fprintf('Inf norm between our solution and CVX solution: %5.3f\n', norm(xCVX - xIP, inf));
%[xIP xCVX]
