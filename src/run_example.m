% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------


function [ yOut ] = run_example( H, z, measurePLQ, processPLQ, params )
%RUN_EXAMPLE Runs simple examples for ADMM comparison
%   Input:
%       H: linear model
%       z: observed data
%    meas: measurement model
%    proc: process model, can be 'none'
%  lambda: tradeoff parameter between process and measurement

t_start = tic;

if(~isfield(params, 'silent'))
   params.silent = 0; 
end
if(~isfield(params, 'constraints'))
   params.constraints = 0; 
end
if(~isfield(params, 'procLinear'))
    params.procLinear = 0;
end
if(~isfield(params, 'proc_scale'))
    params.proc_scale = 1;
end
if(~isfield(params, 'proc_mMult'))
   params.proc_mMult = 1; 
end
if(~isfield(params, 'proc_eps'))
    params.proc_eps = 0.2;
end
if(~isfield(params, 'proc_lambda'))
    params.proc_lambda = 1;
end
if(~isfield(params,'proc_kappa'))
    params.proc_kappa = 1;
end
if(~isfield(params, 'proc_tau'))
    params.proc_tau = 1;
end
if(~isfield(params, 'meas_scale'))
    params.meas_scale = 1;
end
if(~isfield(params, 'meas_mMult'))
   params.meas_mMult = 1; 
end
if(~isfield(params, 'meas_eps'))
    params.meas_eps = 0.2;
end
if(~isfield(params, 'meas_lambda'))
    params.meas_lambda = 1;
end
if(~isfield(params,'meas_kappa'))
    params.meas_kappa = 1;
end
if(~isfield(params, 'meas_tau'))
    params.meas_tau = 1;
end


params.AA = H;
params.b = z;

m = size(params.AA, 1);
par.m = m;

n = size(params.AA, 2);

if(params.procLinear)
    params.pSparse = 0; % should be true if K is sparse
    pLin = params.K;
    k = params.k;
    par.size = length(k);
    par.n = par.size;
else
    params.pSparse = 1;
    pLin = speye(n);
    k = zeros(n,1);
    par.size = n;
    par.n = n;
end



% Define process PLQ
pFlag = 1;
if(isempty(processPLQ))
    pFlag = 0;
end



if(pFlag)
    par.mMult = params.proc_mMult;
    par.lambda = params.proc_lambda;
    par.kappa = params.proc_kappa;
    par.tau = params.proc_tau;
    par.scale = params.proc_scale;
    
    [Mw Cw cw bw Bw] = loadPenalty(pLin, k, processPLQ, par);
end

% Define measurement PLQ

par.size = m;
par.mMult = params.meas_mMult;
par.lambda = params.meas_lambda;
par.kappa = params.meas_kappa;
par.tau = params.meas_tau;
par.eps = params.meas_eps;
par.scale = params.meas_scale;

[Mv Cv cv bv Bv] = loadPenalty(H, z, measurePLQ, par);


%%%%%%

K = size(Bv, 1);
params.pFlag = pFlag;
params.m = m;
params.n = n;
if(pFlag)
    [b, c, C] = addPLQ(bv, cv, Cv, bw, cw, Cw);
    Bm = Bv;
    params.B2 = Bw;
    params.M2 = Mw;
    K = K + size(Bw,1);
else
    b = bv; Bm = Bv; c = cv; C = Cv; M = Mv;
end
C = C';


L = size(C, 2);

sIn = 100*ones(L, 1);
qIn = 100*ones(L, 1);
uIn = zeros(K, 1) + 0.01;
yIn   = zeros(n, 1);

if(params.constraints)
    P = size(params.A, 2);
    rIn = 10*ones(P, 1);
    wIn = 10*ones(P, 1);
else
   rIn = [];
   wIn = [];
end


params.mu = 0;

Fin = kktSystem(b, Bm, c, C, Mv, sIn, qIn, uIn,  rIn, wIn, yIn,params);

[yOut, uOut, qOut, sOut, rOut, wOut, info] = ipSolver(b, Bm, c, C, Mv, sIn, qIn, uIn, rIn, wIn, yIn, params);

Fout = kktSystem(b, Bm, c, C, Mv, sOut, qOut, uOut, rOut, wOut, yOut, params);

ok = norm(Fout) < 1e-6;

fprintf('In, %f, Final, %f, mu, %f, itr %d\n', norm(Fin), norm(Fout), info.muOut, info.itr);

toc(t_start);


end

