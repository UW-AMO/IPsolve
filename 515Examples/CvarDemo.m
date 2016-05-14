cd ..;
clear all;
addpath(genpath(pwd))
cd Scripts;
n = 50;  % number of stocks
m = 100; % time measurements per stock
Y = randn(m, n);  %gaussian samples. 

% TODO: for sasha, check why the hell things fail for larger examples. 


% P seems slow, need to rewrite. and fix Tm, its negative!

YCVX = Y; 
b = zeros(m,1);

% form augmented system 
Yaug = [Y -ones(m,1)]; % for - alpha in objective
YaugCVX = [YCVX -ones(m,1)];
linTerm = [zeros(n,1); 1]; % pick out alpha coordinate
q = m; % this is the number of datapoints
bet = 0.1; % the quantile




params.meas_lambda = 1/(q*(1-bet)); % this is the hinge loss tilt
% here we do the CVX solution
tic
cvx_begin
    variables xCVX(n+1)
    minimize(  xCVX(n+1) + params.meas_lambda*sum(pos(YaugCVX*xCVX)));
    subject to
        ones(1,n)*xCVX(1:n)==1;
        xCVX(1:n)>=zeros(n,1);
cvx_end
toc



% more information for IPsolve. 
params.procLinear = 0;
params.silent = 0;
params.inexact = 0;
params.optTol = 1e-7;
params.progTol = 1e-10;
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

A = [-speye(n); ones(1,n); -ones(1,n)]; % 
A = [A zeros(n+2,1)];
a = [zeros(n,1); 1; -1];  
params.A = A'; 
params.a = a;




[xIP] = run_example(Yaug, b, 'hinge', [], linTerm, params);


'ours, cvx'
[xIP xCVX]

IPobj =  xIP(n+1) + params.meas_lambda*sum(pos(Yaug*xIP));
CVXobj =  xCVX(n+1) + params.meas_lambda*sum(pos(Yaug*xCVX));
diff = (CVXobj - IPobj)/norm(CVXobj);
fprintf('CVX obj: %7.2e, IP obj: %7.2e, rel. diff: %7.2e\n', CVXobj, IPobj, diff);
