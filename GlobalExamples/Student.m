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

m = 200;       % number of examples
n = 50;       % number of features

p = 1;      % sparsity density

x0 = rand(n, 1);
A = randn(m,n);

b = A*x0 + sqrt(0.001)*randn(m,1);
error = 1*sprand(m,1,0.1);
b = b + error;     % add sparse, large noise

params.procLinear = 0;

%Solve problem


params.meas_kappa = 1;
params.uConstraints = 1;
params.meas_scale = 1;
params.uMax = params.meas_scale;
params.uMin = -params.meas_scale;


params.constraints = 0; 
%boxSize = 1;
%Con = [speye(n); -speye(n)];
%params.a = boxSize*ones(2*n, 1);
%params.A = Con';


params.meas_kappa = 1;
[xIP] = run_example(A, b, 'studentPL', [], params);
%Reporting

prm.df = params.meas_scale;
prm.b = b;
prm.A = A;
mFun = @(x)students(x, prm);
xMF = minFunc(mFun, zeros(n,1));


fprintf('Our objective: %5.3f, minFunc objective: %5.3f\n', mFun(xIP), mFun(xMF));

fprintf('Our distance from truth: %5.3f, minFunc distance from truth: %5.3f\n', norm(xIP-x0, inf), norm(xMF-x0, inf));

 fprintf('Inf norm between our solution and minFunc solution: %5.3f\n', norm(xMF - xIP, inf));
 
 sum([abs((xIP-x0)./x0) abs((xMF-x0)./x0)])/n
 
%[xIP xCVX]
