%dir = '/Users/aleksand/Dropbox/QuantileRegression/Mixture/';
%dir = '/Users/aleksand/Dropbox/QuantileRegression/Cauchy/';

%clear all
close all
nonlinear = 1;
probID = 1;


params = [];
if(nonlinear)
    params = getProblemNonlinear(probID, params);
else
    params = getProblemLinear(probID, params);
end

explicit = ~(isa(params.AA,'function_handle'));

if(explicit)
    m = size(params.AA, 1);
    params.m = m;
    n = size(params.AA, 2);
    params.n = n;
else
    m = params.m;
    n = params.n;
end

mu = 10;
%%
H = params.AA;
z = params.b;
params.meas_lambda = 1;
params.proc_lambda = params.lambda;
params.info.pcgIter = [];
params.mehrotra = 1;
params.inexact = 1;
[xIP] = run_example(H, z, 'l1', 'l1', params);



%%
width = 1;
mSize = 10;

figure(1)
plot(1:n, xIP, ':bs', 'LineWidth', width, 'MarkerSize', mSize); 
hold on
plot(1:n, params.x0, '-.r*', 'LineWidth', width)
legend('recovered', 'true');

figure(2)
if(explicit)
    plot(1:m, H*xIP - z, ':bs', 'LineWidth', width)
    hold on
    plot(1:m, H*params.x0 - z, '-.r*', 'LineWidth', width)
else
    plot(1:m, H(xIP) - z,':bs', 'LineWidth', width)
    hold on
    plot(1:m, H(params.x0) - z , '-.r*', 'LineWidth', width)
end
legend('Residual', 'true error');

fprintf('Relative error is %5.4f\n', norm(xIP - params.x0)/norm(params.x0));


