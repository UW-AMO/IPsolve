% L1-regularized least-squares example
%Generate problem data
randn('seed', 0);
rand('seed',0);

smooth = 1;
wellCond = 0;


m = 1000;       % number of examples
n = 2000;       % number of features

p = 100/n;      % sparsity density

x0 = sprandn(n,1,p);
A = randn(m,n);

if(~wellCond)
    [L U] = lu(A);
    ds = speye(m);
    ds = spdiags((1:1:m)', 0, ds);
    U = ds.^20*U;
    A = L*U;    
end

A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns

b = A*x0 + sqrt(0.001)*randn(m,1);
b = b + 10*sprand(m,1,300/m);      % add sparse, large noise

params.procLinear = 0;
params.kappa = 1;
lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;

%Solve problem

if(smooth)
    [xADMM history] = huberl1smooth(A, b, lambda, 1.0, 1.0);
else
    [xADMM history] = huberl1(A, b, lambda, 1.0, 1.0);
end

params.lambda = lambda;
params.kappa = 1;
[xIP] = run_example(A, b, 'huber', 'l1Lam', params);
%Reporting

kappa = cond(A)
accu = 0.5*sum(huber(A*xADMM - b)) + lambda*norm(xADMM,1) - ...
    0.5*sum(huber(A*xIP - b)) - lambda*norm(xIP,1)


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