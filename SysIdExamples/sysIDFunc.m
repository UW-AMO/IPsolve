
function [fRet] = sysIDFunc(z,Q,H,sigma,measurePLQ, processPLQ, params)

L = chol(Q)';

params.pFlag = 1;

explicit = ~(isa(H,'function_handle'));
if(explicit)
    g = run_example( H*L/sigma, z, measurePLQ, processPLQ, params );
else
    fun = @(x) H(L*x/sigma);
    g = run_example( fun, z, measurePLQ, processPLQ, params );
end

fRet = L*g/sigma;

end