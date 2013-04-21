function [ yOut ] = run_example( H, z, measurePLQ, processPLQ, params )
%RUN_EXAMPLE Runs simple examples for ADMM comparison
%   Input:
%       H: linear model
%       z: observed data
%    meas: measurement model
%    proc: process model, can be 'none'
%  lambda: tradeoff parameter between process and measurement

t_start = tic;

if(~isempty(params))
    par.lambda = params.lambda;
end


params.AA = H;
params.b = z;

m = size(params.AA, 1);
par.m = m;
n = size(params.AA, 2);
par.n = n;

% for huber; later this can come in.
par.kappa = 1;

% Define process PLQ
pFlag = 1;
if(isempty(processPLQ))
    pFlag = 0;
end

par.size = n;
if(pFlag)
    [Mw Cw cw bw Bw] = loadPenalty(speye(n), zeros(n,1), processPLQ, par);
end

% Define measurement PLQ

par.size = m;
[Mv Cv cv bv Bv] = loadPenalty(H, z, measurePLQ, par);

K = size(Bv, 1);
par.pFlag = pFlag;
if(pFlag)
    [b, ~, c, C, M] = addPLQ(bw, Bw, cw, Cw, Mw, bv, Bv, cv, Cv, Mv);
    B = Bw;
    par.B2 = Bv;
    K = K + size(B,1);
else
    b = bv; B = Bv; c = cv; C = Cv; M = Mv;
end
C = C';


L = size(C, 2);

sIn = 100*ones(L, 1);
qIn = 100*ones(L, 1);
uIn = zeros(K, 1);
yIn   = zeros(n, 1);

par.mu = 0;
Fin = kktSystem(b, B, c, C, M, sIn, qIn, uIn, yIn, par);

[yOut, uOut, qOut, sOut, info] = ipSolver(b, B, c, C, M, sIn, qIn, uIn, yIn, par);

par.mu = 0;
Fout = kktSystem(b, B, c, C, M, sOut, qOut, uOut, yOut, par);

ok = norm(Fout) < 1e-6;

disp(sprintf('In, %f, Final, %f, mu, %f, itr %d\n', norm(Fin), norm(Fout), info.muOut, info.itr));

toc(t_start);


end

