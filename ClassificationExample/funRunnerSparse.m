addpath(genpath('../'))

n = 100;   % number of data
m = 20;    % size of parameter
problem = 'ls';


A = randn(n,m);  % Gaussian (feature) matrix
theta_init = randn(m,1);      % 'true' values of weights. 

params.A = A;


switch(problem)
    case{'logreg'}
        
        y = sign(A*theta_init + randn(n,1)); % 'true' simulated observations (two classes)
        params.basicFun = @logisticLoss;

        
    case{'ls'}
        params.basicFun = @basicLS;
        y = A*theta_init + .01*randn(n,1);
end

params.y = y;  % this is data
%



params.eps = 0.1; % this is our epsilon in weighted formulation
params.h = n/5;  % h has to be less than $m$, obviously --- h is 'trim size'. 

% make sure initial point is feasible 
wIn = 0.1+rand(n, 1);
if(max(wIn) > 1)
    wIn = wIn/(max(wIn) + .1);
end
params.wIn = wIn;
params.etaIn = 5; % i think this is 'initial' multiplier - algorithm finds the right one




thetaIn = randn(m, 1);  % initial theta.

% 1) First we evaluate: 
%[f g] = functionWeighted(wIn, params);

% 2) Now we do gradient test (check should be 0): 
objFun = @(theta)functionWeighted(theta, params);


%[thetaFinal] = minFunc(objFun, thetaIn, params);

tau = 1;
funProj = @(x)oneProjector(x,1,tau);


thetaFinal = minConf_SPG(objFun,thetaIn,funProj,options);



%thetaFinal = theta_init;

predFinal = A*thetaFinal; % predicted data at final solution
[rFinal resFinal] = params.basicFun(predFinal, y);  % evaluate the function at predicted residual

params.r = rFinal;
params.debugNewton = 0;
params.quiet = 0; 
eta = 1; % initial eta. 

%tic
% here we get the 'final weights' at the final solution - basically testing
% that newton code is working. 
params.debugNewton = 0;
[wFinal eta] = kktNewton(params.wIn, eta, params);
%toc

%%
%SCHUR stuff
Wtilde = eps*diag(wFinal).^(-2) + eps*diag(1-wFinal).^(-2); 
Wtilde(Wtilde == inf) = 0;

hess = [ A'*diag(wFinal)*A A'*diag(rFinal); diag(rFinal)*A Wtilde  ];

denom = eps*(1-wFinal).^2 + eps*(wFinal.^2);
r = params.r;
Schur = A'*( eye(n) - diag(r.^2.*wFinal.^2.*(1-wFinal).^2./denom) )*A;

% 
% minimal eigenvalue
%eigs(Schur, 1, 'SM')
condSchur = cond(Schur);

fprintf('\n Condition of Schur complement is: %5.3f\n ', condSchur);





% params.debugNewton = 0; 
% params.quiet = 1; 
% 
% params.r = r; 
% [wFinal eta] = kktNewton(wIn, params.etaIn, params);

