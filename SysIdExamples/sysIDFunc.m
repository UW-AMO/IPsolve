
function [fRet] = sysIDFunc(z,Q,H,sigma,measurePLQ, processPLQ, params)

L = chol(Q)';

params.pFlag = 1;

g = run_example( H*L/sigma, z, measurePLQ, processPLQ, params );

fRet = L*g/sigma;

end