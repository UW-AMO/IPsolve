function [ ok ] = plotPenalty( plq, params )
%PLOTPENALTY plots a given PLQ penalty
%   plq:    name of penalty. If it's in the library, the system will generate 
%           a 2D plot. 
%   params: structure containing parameters, 
%           e.g. tau (quantile)
%                lambda(quantile, l1)
%                kappa (huber, quantile huber). 

H = 1; 
z = 0;
K = 1;
params.size = 1;
params.m = 1;
n = 1;
params.n = 1;
P = 2; % constraints

if(~isfield(params, 'pSparse'))
   params.pSparse = 1; 
end
if(~isfield(params, 'pFlag'))
   params.pFlag = 0; 
end
if(~isfield(params, 'silent'))
   params.silent = 1; 
end
if(~isfield(params, 'procLinear'))
    params.procLinear = 0;
end
if(~isfield(params, 'mMult'))
   params.mMult = 1; 
end
if(~isfield(params, 'lambda'))
    params.lambda = 1;
end
if(~isfield(params,'kappa'))
    params.kappa = 1;
end
if(~isfield(params, 'tau'))
    params.tau = 1;
end

[ M C c b B ] = loadPenalty( H, z, plq, params );
C = C'; % this is funny. 
L = size(C, 2);
sIn = 100*ones(L, 1);
qIn = 100*ones(L, 1);
uIn = zeros(K, 1);
rIn = 100*ones(P, 1);
wIn = 100*ones(P, 1);

params.constraints = 1;
params.A = [1, -1];

mus = -3:.05:3;
len = length(mus);    
vals = zeros(len,1);

for i = 1:len
    params.a = [mus(i);-mus(i)]; % constrains x = \mu. 
    yIn = mus(i);
    [yOut, uOut, ~, ~, ~, ~, ~] = ipSolver(b, B, c, C, M, sIn, qIn, uIn, rIn, wIn, yIn, params);
%    fprintf('yOut value us: %5.3f\n', yOut);
    vals(i) = uOut'*B*yOut - 0.5*uOut'*M*uOut;
end


plot(mus, vals);
ok = 1;

end

