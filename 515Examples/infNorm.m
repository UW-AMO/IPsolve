% set up classic compressive sensing problem 
clear all ;close all; clc
  m = 500; n = 4000; k = 100; % m rows, n cols, k nonzeros.
  p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
  A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
  b  = A*x0 + 0.005 * randn(m,1);

  lam = norm(A'*b, 'inf')/10; 
  
  %% IP solve: full functionality for lasso problem 
  params.proc_lambda = lam;
  params.inexact = 0;
  params.simplex = 1;
  params.optTol = 1e-7;
  xLasso = run_example( A, b, 'l2', 'infnorm', [], params );
  % careful, assumes the simplex rows are coming first in the C... 
  
  plot(1:n, x0, 1:n, xLasso)
  legend('truth', 'IPsolve solution')
  
  
