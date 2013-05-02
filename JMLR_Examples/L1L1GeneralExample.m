% Written by Aleksandr Aravkin (saravkin at us dot ibm dot com)
%General L1 + L1 example

%Generate problem data

randn('seed', 0);
rand('seed',0);

m = 1000;       % number of examples
n = 2000;       % number of features
k = 500;        % extra regularizer. 

p = 100/n;      % sparsity density

x0 = sprandn(n,1,p);
A = randn(m,n);
K = randn(k,n);
%A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
b = A*x0 + sqrt(0.001)*randn(m,1);
b = b + 10*sprand(m,1,200/m);      % add sparse, large noise


lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;

%lambda = 1;

%Solve problem



params.lambda = lambda;
params.kappa = 1;
params.K = K;
params.k = zeros(k,1);
params.procLinear = 1;
[xIP] = run_example(A, b, 'huber', 'l1', params);
%Reporting


%fprintf('Solution inf norm difference: %f\n', norm(xIP - xADMM, inf)); 


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