cd ..;
clear all;
addpath(genpath(pwd))
cd Scripts;
n = 50;  % number of stocks
m = 100; % time measurements per stock
Y = randn(m, n);  %gaussian samples. 

% TODO: for sasha, check why the hell things fail for larger examples. 

useP = 0; 
E = ones(n,1); % the E from the writeup
P = @(x,mode) x - mean(x); 
P = opFunction(n, n, P);
%P = opEye(n) - opOnes(n,n)/n;

% P seems slow, need to rewrite. and fix Tm, its negative!


YCVX = Y; 
if(useP)
    Y = Y*P;
    b = Y*E/n;
else
    b = zeros(m,1);
end
% form augmented system 
Yaug = [Y -ones(m,1)]; % for - alpha in objective
YaugCVX = [YCVX -ones(m,1)];
linTerm = [zeros(n,1); 1]; % pick out alpha coordinate
q = m; % this is the number of datapoints
bet = 0.1; % the quantile




params.meas_lambda = 1/(q*(1-bet)); % this is the hinge loss tilt
% here we do the CVX solution
cvx_begin
    variables xCVX(n+1)
    minimize(  xCVX(n+1) + params.meas_lambda*sum(pos(YaugCVX*xCVX)));
    subject to
        ones(1,n)*xCVX(1:n)==1;
        xCVX(1:n)>=zeros(n,1);
cvx_end




% more information for IPsolve. 
params.procLinear = 0;
params.silent = 0;
params.inexact = 0;
params.optTol = 1e-8;
params.progTol = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% constarints interface to IPsolve
% params.constraints = 1; 
% A = [-speye(n); ones(1,n); -ones(1,n)]; % contraint matrix
% A = [A zeros(n+2,1)]; 
% a = [zeros(n,1); 1; -1]; % cosntraint vector 
% params.A = A'; 
% params.a = a;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

params.constraints = 1; 

if(useP)
    A = -P; % contraint matrix
    A = [A zeros(n,1)];
    a = E/n; % cosntraint vector
else
    A = [-speye(n); ones(1,n); -ones(1,n)]; % contraint matrix
    A = [A zeros(n+2,1)];
    a = [zeros(n,1); 1; -1]; % cosntraint vector
end
    
params.A = A'; 
params.a = a;



% regularization parameters, not usually useful
if(useP)
    params.rho = 0;
    params.delta = 1e-5;
else
    params.rho = 0;
    params.delta = 0; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%
%[xIP] = run_example(Yaug, zeros(m,1), 'hinge', [], linTerm, params);

% now with new map! 
if(useP)
    [uIP] = run_example(Yaug, b, 'hinge', [], linTerm, params);
    xIP = [P*uIP(1:end-1)+ E/n; uIP(end)]; % remake the solution
else
    [xIP] = run_example(Yaug, b, 'hinge', [], linTerm, params);
end

'ours, cvx'
[xIP xCVX]
%[xIP] = run_example(A, b, 'l2', 'l1', params);
%Reporting



% generate a matrix