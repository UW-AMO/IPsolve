% Least absolute deviations example
%Generate problem data

rand('seed', 0);
randn('seed', 0);

m = 5000; % number of examples
n = 200;  % number of features

A = randn(m,n);
x0 = 10*randn(n,1);
b = A*x0;
idx = randsample(m,ceil(m/50));
b(idx) = b(idx) + 1e2*randn(size(idx));

%Solve problem

[xADMM history] = lad(A, b, 1.0, 1.0);

[xIP] = run_example(A, b, 'l1', [], []);

fprintf('Solution norm difference: %f\n', norm(xIP - xADMM)); 


%Reporting

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