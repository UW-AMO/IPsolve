clear all; close all; clc

cd ..
addpath(genpath(pwd))
cd 515Examples
% set up a linear system and parameter vector
m = 1000;
n = 700; 
A = randn(m,n); 
x0 = randn(n,1); 

% noise and outlier parameters
sig = .05; % gaussian noise variance
out = 30;  % number of outliers 
mag = 20;   % outlier variance

% construct the outliers
outliers = zeros(m,1);
inds = randperm(m, out); % indices of outliers
outliers(inds) = mag*randn(out,1); 

% construct measured data b
b0 = A*x0; 
b = b0 + .05*randn(m,1) + outliers; 

% define a relative error function
errFunc = @(x) norm(x-x0)/norm(x0);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Least squares and ridge regression solutions
lam = 1; % ridge regularizer
xLS = (A'*A)\A'*b; 
xRidge = (A'*A + lam*eye(n))\(A'*b);

errLS       = errFunc(xLS);
errRidge    = errFunc(xRidge);

fprintf('LS error: %5.2e, Ridge error: %5.2e\n', errLS, errRidge);

%% running L1 huber regression using full functionality of IPsolve

params.proc_lambda = lam;
params.proc_mMult = lam;

params.meas_kappa = 0.05;
xHuber =  run_example( A, b, 'huber', [], [], params );
errHuber = errFunc(xHuber);

fprintf('LS error: %7.1e, Huber error: %7.1e\n', errLS, errHuber);




% %% Adding regularization
% xLasso =  run_example( A, b, 'l2', 'l1', [], params );
% xHuberRidge = run_example( A, b, 'huber', 'l2', [], params );
% xHuberLasso = run_example( A, b, 'huber', 'l1', [], params );
% 
% errLasso = errFunc(xLasso);
% errHuberRidge = errFunc(xHuberRidge);
% errHuberLasso = errFunc(xHuberLasso);
% 
% fprintf('LS error: %7.1e, Ridge error: %7.1e, Lasso error: %7.1e\n', errLS, errRidge, errLasso);
% 
% fprintf('Huber error: %7.1e, Huber Ridge error: %7.1e, Huber Lasso error: %7.1e\n', errHuber, errHuberRidge, errHuber);

%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % huber regression using direct matrix construction + solver
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kappa = 1; % huber parameter
% B = A; 
% b = -b; 
% C = [speye(m); -speye(m)]; 
% c = kappa*[ones(2*m,1)];
% M = speye(m); 
% 
% % initialize q, u, r, w, x
% r = []; % no constraints
% w = []; % no constraints
% linTerm = 0; % no linear term
% C = C';  % artifact of code
% [K,L] = size(C);
% q = 10*ones(L, 1);
% u = zeros(K, 1) + .01;
% x   = ones(n, 1);
% 
% 
% % set up parameters
% params.n = n; % dimension of vector 
% params = setParms(params, 1); % sets lots of parameters 
% 
% fun1 = @(x) sum((abs(x) > kappa).*(abs(x) - 0.5*kappa) + 0.5*(abs(x) <= kappa).*x.^2/(kappa)); 
%        % for iteration dispay only
% params.objFun = @(x)fun1(A*x-b); 
% 
% % direct call to solver - note sign change on b, and transpose on C
% 
% % [yOut, uOut, qOut, rOut, wOut, info] = ipSolverBarrier(linTerm, b, Bm, c, C, M, q, u, r, w, y, params)
% [xHuber, uOut, qOut, ~, ~, ~] = ipSolverBarrier(linTerm, b, B, c, C, M, q, u, r, w, x, params);
% errHuber = errFunc(xHuber);
% fprintf('LS error: %5.2e, Ridge error: %5.2e, Huber error: %5.2e\n', errLS, errRidge, errHuber);
% 
% 
% %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Ridge huber regression using direct matrix construction + solver
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Huber regression alone Using automated penalty loading
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %[ M C c b B fun] = loadPenalty( H, z, penalty, params )
% par.kappa = 1; % huber penalty parameter
% par.mMult = 1; % curvature coefficient
% par.size = m;  % size of the vector (in this case, residuals) 
% 
% % careful with the b!!
% [M1, C1, c1, b1, B1, fun2] = loadPenalty(A, -b, 'huber', par);
% 
% C1 = C1';
% 
% %reinitialize to be safe
% [K,L] = size(C1);
% q = 10*ones(L, 1);
% u = zeros(K, 1) + .01;
% x   = ones(n, 1);
% 
% params.objFun = @(x)fun2(A*x-b); % this is the function we had to manually program before
% 
% % calling solver again: 
% errM = norm(M-M1, 'fro');
% errC = norm(C1 - C, 'fro'); 
% errc = norm(c1 - c, 'fro');  
% errB = norm(B1 + B, 'fro');  % artifact of interface 
% errObjFun = fun1(x0) - fun2(x0);
% fprintf('discrepancies: M %5.2e, C: %5.2e, c: %5.2e, B: %5.2e, fun: %5.2e\n', errM, errC, errc, errB, errObjFun);
% [xHuber, uOut, qOut, ~, ~, ~] = ipSolverBarrier(linTerm, b1, B1, c1, C1, M1, q, u, r, w, x, params);
% errHuber = errFunc(xHuber);
% fprintf('LS error: %5.2e, Ridge error: %5.2e, Huber error: %5.2e\n', errLS, errRidge, errHuber);
% 
% %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Huber + L1 regression alone Using automated penalty loading
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % first load huber for residuals
% par.kappa = 1; % huber penalty parameter
% par.mMult = 1; % curvature coefficient
% par.size = m;  % size of the vector (in this case, residuals) 
% [Mh, Ch, ch, bh, Bh, funH] = loadPenalty(A, -b, 'huber', par);
% 
% % next load l1 for x coefficients
% % first load huber for residuals
% par.lambda = 1; % slope parameter (1-norm regression)
% par.size = n;  % size of the vector (in this case, coefficients) 
% [Ml, Cl, cl, bl, Bl, funL] = loadPenalty(speye(n), zeros(n,1), 'huber', par);
% 
% 
% % now form the SUM PLQ: 
% [bfull, Bfull, cfull, Cfull, Mfull] = addPLQFull(bh, Bh, ch, Ch, Mh, bl, Bl, cl, Cl, Ml);
% % and make the net objective function:
% params.objFun = @(x) funH(A*x-b) + funL(x); 
% 
% 
% % initialize q, u, r, w, x
% r = []; % no constraints
% w = []; % no constraints
% linTerm = 0; % no linear term
% [L,K] = size(Cfull); % need to update 
% q = 10*ones(L, 1);
% u = zeros(K, 1) + .01;
% x   = ones(n, 1);
% 
% [xHuberL1, uOut, qOut, ~, ~, ~] = ipSolverBarrier(linTerm, bfull, Bfull, cfull, Cfull', Mfull, q, u, r, w, x, params);
% errHuberL1 = errFunc(xHuberL1);
% fprintf('LS error: %5.2e, Ridge error: %5.2e, Huber error: %5.2e, L1 Huber Error: %5.2e\n', errLS, errRidge, errHuber, errHuberL1);


