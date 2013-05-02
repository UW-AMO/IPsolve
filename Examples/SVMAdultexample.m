
%Linear support vector machine example

rand('seed', 0);
randn('seed', 0);

[y,x] = libsvmread('./adult/a9a.trn');

n = size(x, 2);
m = size(x, 1);

Y = repmat(y, 1, n);
A =[ Y.*x y];
%A = [ -x -y];   % 200 x 3

lambda = 1.0;


[xADMM history] = linear_svm(A, lambda, 1.0, 1.0);

wADMM = xADMM(1:end-2);
bADMM = xADMM(end);

params.procLinear = 0;
params.lambda = lambda;
[xIP] = run_example(-A, -ones(m,1), 'hinge', 'l2m', params);


[yTest, xTest] = libsvmread('./adult/a9a.t');

wIP = xIP(1:end-2);
bIP = xIP(end);

yProp = sign(xTest*wADMM - bADMM);
errADMM = sum(yTest ~= yProp)/length(yProp)

yProp = sign(xTest*wIP - bIP);
errIP = sum(yTest ~= yProp)/length(yProp)
%fprintf('Solution norm difference: %f\n', norm(xIP - xADMM)); 


kappa = sqrt(cond(A'*A))

accu = 0.5*norm(xADMM(1:end-1))^2 + lambda*sum(pos(1-A*xADMM ))-...
0.5*norm(xIP(1:end-1))^2 - lambda*sum(pos(1-A*xIP))


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