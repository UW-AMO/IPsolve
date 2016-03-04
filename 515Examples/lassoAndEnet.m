% set up classic compressive sensing problem 
clear all ;close all; clc
  m = 120; n = 512; k = 20; % m rows, n cols, k nonzeros.
  p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
  A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
  b  = A*x0 + 0.005 * randn(m,1);

  lam = norm(A'*b, 'inf')/10; 
  
  %% IP solve: full functionality for lasso problem 
  params.proc_lambda = lam;
  xLasso = run_example( A, b, 'l2', 'l1', [], params );
  
  plot(1:n, x0, 1:n, xLasso)
  legend('truth', 'IPsolve solution')
  
  
  %% manually building up all the penalties 
  
  % first load huber for residuals
par.mMult = 1; 
par.size = m;  % size of the vector (in this case, residuals)
[Msq, Csq, csq, bsq, Bsq, funSq] = loadPenalty(A, b, 'l2', par);

par.lambda = lam; 
par.size = n;  % size of the vector (in this case, x)
[Ml1, Cl1, cl1, bl1, Bl1, funL1] = loadPenalty(speye(n), zeros(n,1), 'l1', par);

% now form the SUM PLQ: 
[bfull, Bfull, cfull, Cfull, Mfull] = addPLQFull(bsq, Bsq, csq, Csq, Msq, bl1, Bl1, cl1, Cl1, Ml1);
% and make the net objective function:
params.objFun = @(x) funSq(A*x-b) + funL1(x); 

r = []; % no constraints
w = []; % no constraints
linTerm = 0; % no linear term
[L,K] = size(Cfull); % need to update 
q = 10*ones(L, 1);
u = zeros(K, 1) + .01;
x   = ones(n, 1);

% sets lots of parameters 
params.n = n;
params = setParms(params, 1); 

[xLasso2, uOut, qOut, ~, ~, ~] = ipSolverBarrier(linTerm, bfull, Bfull, cfull, Cfull', Mfull, q, u, r, w, x, params);

normErr = norm(xLasso2 - xLasso)/norm(xLasso);
fprintf('discrepancy: %5.2e\n', normErr);
  

%% now, build up a problem to solve elastic net: 
% min ||Ax-b||^2 + 
%  \lambda * \alpha * ||x||_1 + \lambda (1-\alpha) ||x||^2/2
% where lambda and alpha are given by the user. 







%% If your approach worked, try modifying loadPenalty to add an elastic net
% penalty. this will allow you to run a command such as 
%  xLasso = run_example( A, b, 'l2', 'enet', [], params );


