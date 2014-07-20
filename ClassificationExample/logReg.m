% Written by Aleksandr Aravkin (saravkin at us dot ibm dot com)
% L1-regularized Huber example. 
%clear all;
%Generate problem data
randn('seed', 0);
rand('seed',0);
%clear all; close all; clc;
warning off all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% m = 10000;       % number of examples
% n = 1000;       % number of features
% 
% 
% x0 = randn(n, 1);
% A = [ones(m,1) randn(m,n-1)];
% 
% b = sign(A*x0 + sqrt(0.01)*randn(m,1));
% 
% Adat = diag(b)*A;
% %b = b + 10*sprand(m,1,300/m);      % add sparse, large noise


fprintf('Loading Data\n');
%load('Data/rcv1_train.binary.mat'); data_flag = 1; dataset_name = 'rcv1';
%load('adult.mat'); data_flag = 2; dataset_name = 'adult';
load('sido.mat'); data_flag = 3; dataset_name = 'sido';
X = [ones(size(X,1),1) X];
[size_dataset,p] = size(X);

perm_index = randperm(size(X,1));
X_train = X(perm_index(1:ceil(0.9*size(X,1))),:);
y_train = y(perm_index(1:ceil(0.9*size(X,1)))); 
X_test = X(perm_index(ceil(0.9*size(X,1))+1:end),:);
y_test = y(perm_index(ceil(0.9*size(X,1))+1:end));

%load('data_debug.mat');
% load('bupa.mat');
% data_flag = 1; 
% dataset_name = 'bupa';

X = X_train;
y = y_train;
Xt = X';

params.Anorm = normest(Xt);

Y = speye(length(y));

Y = spdiags(y,0,Y);

Xlab = Y*X;
Xlabt = Xlab';
[n,p] = size(X);


params.procLinear = 0;
params.uConstraints = 1;
params.meas_scale = 1;
params.uMax = 1;
params.uMin = 0; % 



 params.constraints = 0;
 if(params.constraints)
     boxSize = 1;
     Con = [speye(p); -speye(p)];
     params.a = boxSize*ones(2*p, 1);
     params.A = Con';
 end
 
 
 params.inexact = 1;
 params.proc_lambda = 1e1;
% params.proc_mMult = 1e4;
 params.rho = 0;
 params.delta = 0;
[xIP] = run_example(Xlab, 0*y, 'logreg', 'l1', params);



