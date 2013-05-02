% Distributed linear support vector machine example
%Generate problem data

rand('seed', 0);
randn('seed', 0);

n = 2;
m = 100000;
N = m/2;
M = m/2;

% positive examples
Y = [1.5+0.9*randn(1,0.6*N), 1.5+0.7*randn(1,0.4*N);  % 60% of them are Gaussian mean 1.5 std 0.9,  40% of them are Gaussian mean 1.5 std 0.7
2*(randn(1,0.6*N)+1), 2*(randn(1,0.4*N)-1)];          % 60% of them are Gaussian mean 2   std 2,    40% of them are Gaussian mean -2  std 2

% negative examples
X = [-1.5+0.9*randn(1,0.6*M),  -1.5+0.7*randn(1,0.4*M); % 60% of them are Gaussian mean 1.5 std 0.9,  40% of them are Gaussian mean 1.5 std 0.7
2*(randn(1,0.6*M)-1), 2*(randn(1,0.4*M)+1)];            % 60% of them are Gaussian mean 2   std 2,    40% of them are Gaussian mean -2  std 2

x = [X Y];
%x = [X Y ; X Y; X Y ; X Y];                  % all examples (2x200)  where the 200 is 100 positive and 100 negative, the 2 is train/test?? NO I THINK THE INPUTS ARE JUST 2 DIMENSIONAL  
y = [ones(1,N) -ones(1,M)]; % true labels
A = [ -((ones(n,1)*y).*x)' -y'];   % 200 x 3
xdat = x';
lambda = 1.0;

% partition the examples up in the worst possible way
% (subsystems only have positive or negative examples)
%p = zeros(1,m);  % zeros(size(y))
%p(y == 1)  = sort(randi([1 10], sum(y==1),1));    % generate random integers between 1  and 10. the number of random integers equals the number of positive examples. then sort
%p(y == -1) = sort(randi([11 20], sum(y==-1),1));  % generate random integers between 11 and 20. the number of random integers equals the number of negative examples. then sort
% the above 3 lines creates 20 partitions of the data. the partition
% allocations are stored in p. the partitions are created such that each of
% the 20 partitions contains either all positive examples or all negative
% examples. the 20 partitions do not necessarily have the same number of
% members but each has, on average, 1/20 of the data

% ONE PARTITION ONLY -- WE DON'T CARE ABOUT DISTRIBUTED SVMS RIGHT NOW 
p = ones(1,m);

%Solve problem

%[xADMM history] = linear_svm(A, lambda, p, 1.0, 1.0);

params.procLinear = 0;
params.lambda = lambda;
[xIP] = run_example(A, -ones(m,1), 'hinge', 'l2m', params);

%fprintf('Solution norm difference: %f\n', norm(xIP - xADMM)); 



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