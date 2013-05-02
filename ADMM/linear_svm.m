% Written by Aleksandr Aravkin (saravkin at us dot ibm dot com)
% Overal development and structure inherited from methods found here: 
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html



function [x, history] = linear_svm(A, lambda, rho, alpha)
% linear_svm  Solve linear SVM problem via ADMM
%
% [z, history] = lasso(A, b, lambda, rho, alpha);
%
% Solves the following problem via ADMM:
%
%   minimize 1/2*||w||_2^2 + lambda hinge ( Ax - 1), x = [w b].
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
MAX_ITER = 1000;
ABSTOL   = 1e-6;
RELTOL   = 1e-4;

%Data preprocessing

[m, n] = size(A);

% save a matrix-vector multiply
%Atb = A'*b;

%ADMM solver

x = zeros(n,1);
z = zeros(m,1);
u = zeros(m,1);

% cache the factorization
[L U] = factor(A, rho);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER

    % x-update

    %obj1 = 0.5*norm(x(1:end-1))^2 + 0.5*rho*norm(A*x - z - u - 1)^2
    q = rho*A'*(1-z-u);    % temporary value
    if( m >= n )    % if skinny
       x = U \ (L \ q);
    else            % if fat
       x = q/rho - (A'*(U \ ( L \ (A*q) )))/rho^2;
    end
   % obj2 = 0.5*norm(x(1:end-1))^2 + 0.5*rho*norm(A*x - z - u - 1)^2


    % z-update with relaxation
    zold = z;
    q = 1 - A*x - u;
%    qm = -pos(-q);
%    qp = pos(q);
    z = shrinkagePos(q, lambda/rho);

    % u-update
    u = u + (A*x + z - 1);

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, lambda, x);

    history.r_norm(k)  = norm(A*x + z - 1);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k)) && k 
         break;
    end

end

if ~QUIET
    toc(t_start);
end
end

function p = objective(A,lambda, x)
    p = 0.5*norm(x(1:end-1))^2 + lambda*sum(pos(1-A*x));
end

function z = shrinkagePos(x, kappa)
    z = max( 0, x - kappa ) - max( 0, -x);
end

function [L U] = factor(A, rho)
    [m, n] = size(A);
    sp = speye(n);
    sp(1) = 0;
    if ( m >= n )    % if skinny
       L = chol( rho*(A'*A) + sp, 'lower' );
    else            % if fat
       L = chol( 1/rho*sp + A*A', 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end