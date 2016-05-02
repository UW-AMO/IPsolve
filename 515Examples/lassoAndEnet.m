% set up classic compressive sensing problem 
clear all ;close all; clc

wellCond = 1;

  m = 120; n = 512; k = 10;
  %m = 1200; n = 5120; k = 100; % m rows, n cols, k nonzeros.
  p = randperm(n); x0 = zeros(n,1); x0(p(1:k)) = sign(randn(k,1));
  A  = randn(m,n); 
  
  if(wellCond)
      [Q,R] = qr(A',0);  A = Q';
  end
  b  = A*x0 + 0.005 * randn(m,1);
  
  nrmA = normest(A);
  fprintf('Norm of A: %7.1e\n', nrmA); 

  lam = norm(A'*b, 'inf')/10; 
  
  %% IP solve: full functionality for lasso problem 
  params.proc_lambda = lam;
  params.inexact = 0;
  xLasso = run_example( A, b, 'l2', 'l1', [], params );
  
  plot(1:n, x0, 1:n, xLasso)
  legend('truth', 'IPsolve solution')
  
  
