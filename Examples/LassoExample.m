% L1-regularized least-squares example
%Generate problem data

randn('seed', 0);
rand('seed',0);

m = 300;       % number of examples
n = 5000;       % number of features
p = 100/n;      % sparsity density

x0 = sprandn(n,1,p);
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
b = A*x0 + sqrt(0.001)*randn(m,1);

lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;

%Solve problem
[xADMM history] = lasso(A, b, lambda, 1.0, 1.0);

params.procLinear = 0;

params.lambda = lambda;
[xIP] = run_example(A, b, 'l2', 'l1Lam', params);
%Reporting

accu = 0.5*norm(A*xADMM - b)^2 + lambda*norm(xADMM,1) - ...
0.5*norm(A*xIP - b)^2 - lambda*norm(xIP,1)  


fprintf('Solution inf norm difference: %f\n', norm(xIP - xADMM, inf)); 


% K = length(history.objval);
% 
% h = figure;
% plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
% ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');
% 
% g = figure;
% subplot(2,1,1);
% semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
%     1:K, history.eps_pri, 'k--',  'LineWidth', 2);
% ylabel('||r||_2');
% 
% subplot(2,1,2);
% semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
%     1:K, history.eps_dual, 'k--', 'LineWidth', 2);
% ylabel('||s||_2'); xlabel('iter (k)');