function [x_CVaR, x_RCVaR]= RobustCVaR(Y, beta, rho) 

%%% Based on the idea described in RobusrCVaR.pdf
% still not quite convinced that this problem is equivalent of the
% original problem. Will check the results.


%% initialize parameters
rng(1,'twister'); % set random seed and generator
n = size(Y,2);  % number of stocks
m = size(Y,1); % time measurements per stock
% Y = randn(m, n);  %gaussian samples of losses
% goodcolumn=1;
% badcell=10;
% Y(:,goodcolumn)=Y(:,goodcolumn)-0.2;
% Y(badcell,goodcolumn)=Y(badcell,goodcolumn)+15;
% q=m;
%beta=0.9;  % The lower tail probability
%rho=0.99; % The higher tail probability


maxiter=1e1; % maximum number of iterations

x0=ones(n,1)./n;
x=x0;


%% Solve the problem iteratively
for iter=1:maxiter
    % fint the w's corresponding to the current x
    % by finding the indices of the assets that has a loss
    % between the lower and upper quantile
    port_losses=Y*x;
    XIs = port_losses;
    q_lower=-Inf;
    q_upper=quantile(port_losses,rho);
    w=zeros(m,1);
    tmp_idx=(port_losses>=q_lower)&(port_losses<=q_upper);
    w(tmp_idx)=1;
    
    % now the new loss vector is w'*port_losses    
% The subproblem is a convex problem. Solve using CVX
Yaug = [Y -ones(m,1)]; % for - alpha in objective
linTerm = [zeros(n,1); 1]; % pick out alpha coordinate
params.meas_lambda = 1/(m*(rho-beta)); % this is the hinge loss tilt
% here we do the CVX solution
cvx_begin quiet
    variables xCVX(n+1)
    minimize(  xCVX(n+1) + params.meas_lambda*sum(w'*pos(Yaug*xCVX)));
    subject to
        ones(1,n)*xCVX(1:n)==1;
        xCVX(1:n)>=zeros(n,1);
cvx_end
x=xCVX(1:n);
alpha = xCVX(n+1);


end
x_RCVaR = x; 


%% Solve the problem iteratively
for iter=1:1
    % fint the w's corresponding to the current x
    % by finding the indices of the assets that has a loss
    % between the lower and upper quantile
    port_losses=Y*x;
    q_lower=-Inf;
    q_upper=Inf;
    w=ones(m,1);
    
    % now the new loss vector is w'*port_losses    
% The subproblem is a convex problem. Solve using CVX
Yaug = [Y -ones(m,1)]; % for - alpha in objective
linTerm = [zeros(n,1); 1]; % pick out alpha coordinate
params.meas_lambda = 1/(m*(1-beta)); % this is the hinge loss tilt
% here we do the CVX solution
cvx_begin quiet
    variables xCVX(n+1)
    minimize(  xCVX(n+1) + params.meas_lambda*sum(w'*pos(Yaug*xCVX)));
    subject to
        ones(1,n)*xCVX(1:n)==1;
        xCVX(1:n)>=zeros(n,1);
cvx_end
x_CVaR=xCVX(1:n);


end

%[x_CVaR x_RCVaR]




