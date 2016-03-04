function params = setParms(params, explicit)

if(explicit)
    params.relTol = 0;  % solve each subproblem to completion
else
    params.relTol = 0.1; % solve each subproblem approximately.
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% general algorithm parameteres

if(~isfield(params, 'pFlag'))
    params.pFlag = 0;
end

if(~isfield(params, 'pSparse'))
    params.pSparse = 1; % this is ok, since run_example modifes it anyway
end

if(~isfield(params, 'relOpt'))
    params.relOpt = 1e-5;
end
if(~isfield(params, 'optTol'))
    params.optTol = 1e-5;
end
if(~isfield(params, 'silent'))
    params.silent = 0;
end
if(~isfield(params, 'constraints'))
    params.constraints = 0;
end
if(~isfield(params, 'inexact'))
    params.inexact = 0;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% controls for process model

if(~isfield(params, 'dictionary')) % specific to dictionary learning
    params.dictionary = 0;
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

if(~isfield(params,'rho'))
    params.rho = 0;
end
if(~isfield(params, 'delta'))
    params.delta = 0;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% control for measurement model
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% control for restricting conjugate domain
if(~isfield(params, 'uConstraints'))
    params.uConstraints = 0;
end

% control for using predictor-corrector
if(~isfield(params, 'mehrotra'))
    params.mehrotra = 0;
end



end