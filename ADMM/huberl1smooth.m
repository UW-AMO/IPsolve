% Written by Aleksandr Aravkin (saravkin at us dot ibm dot com)
% Overal development and structure inherited from methods found here: 
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html

% ATTENTION! 
% This file requires the package minFunc, which is found here: 
% http://www.di.ens.fr/~mschmidt/Software/minFunc.html

% Alternatively, minFunc can be replaced with your favorite routine 
% that requires only function values and gradients. 


function [x, history] = huberl1smooth(A, b, lambda, rho, alpha)
% lasso  Solve lasso problem via ADMM
%
% [z, history] = lasso(A, b, lambda, rho, alpha);
%
% Solves the following problem via ADMM:
%
%   minimize hub(Ax - b ) + \lambda || x ||_1
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% alpha is the over-relaxation parameter (typical values for alpha are
% between 1.0 and 1.8).
%
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;
%Global constants and defaults

QUIET    = 0;
MAX_ITER = 100;
ABSTOL   = 1e-5;
RELTOL   = 1e-3;

%Data preprocessing

[m, n] = size(A);

% save a matrix-vector multiply
%Atb = A'*b;

%ADMM solver

x = zeros(n,1);
z = zeros(n,1);
y = zeros(n,1);

% cache the factorization
%[L U] = factor(A, rho);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

options.Display = 'off';
for k = 1:MAX_ITER

%    x = lasso_fact(A, L, U, b+z-y, lambda/rho, rho, alpha); 
    params.hub = 1;
    params.rho = rho;
    func = @(x)funHUBL2(A, b, x, z-y, params);
    x = minFunc(func, x, options);
    % x-update
    %q = Atb + rho*(z - y);    % temporary value
    
    % z-update with relaxation
    zold = z;
%    x_hat = alpha*x + (1 - alpha)*zold;
%    x_hat = x;

%    z = shrinkage(A*x-b-y, 1/rho); % maybe?
  %  tmp = A*x - b + y;
%    z = rho/(1 + rho)*tmp + 1/(1 + rho)*shrinkage(tmp, 1 + 1/rho);
    z = shrinkage(x+y, lambda/rho); % maybe?


    % u-update
    y = y + (x - z);

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, b, lambda, x);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(A*x-b), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*y);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end

end

if ~QUIET
    toc(t_start);
end
end

function p = objective(A, b, lambda, x)
    p = 0.5*sum(huber(A*x - b)) + lambda*norm(x,1);
end

function z = shrinkage(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x - kappa );
end

function [L U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( A'*A + rho*speye(n), 'lower' );
    else            % if fat
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end