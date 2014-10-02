%dir = '/Users/aleksand/Dropbox/QuantileRegression/Mixture/';
%dir = '/Users/aleksand/Dropbox/QuantileRegression/Cauchy/';

probID = 9;


params = [];
params = getProblemLinear(probID, params);

m = size(params.AA, 1);
params.m = m;
n = size(params.AA, 2);
params.n = n;

mu = 10;
%%
H = params.AA;
z = params.b;
params.meas_lambda = 1;
params.proc_lambda = .1;

opts          = as_setparms;
opts.loglevel = 1;
inform        = [];  % IMPORTANT: must initialize in this way.

% Re-solve the BPDN problem: minimize .5||Ax-b||^2 + lambda||x||_1


% run ASP
tic
[x,inform] = as_bpdn(H,z,params.proc_lambda,opts,inform);
toc     
% run IP
[xIP] = run_example(H, z, 'l2', 'l1', params);

fprintf('norm of difference: %5.3f\n', norm(x - xIP, inf));

plot(1:n, x, 1:n, xIP + 2);


%%


%figure(1)
%plot(1:n, yOut-1, 1:n, trueBeta + 1)

%figure(2)
%plot(1:m, H*trueBeta - z + 25, 1:m, H*yOut - z - 25)
