% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------


function [ yOut normFout] = run_example( H, z, measurePLQ, processPLQ, params )
%RUN_EXAMPLE Runs simple examples for ADMM comparison
%   Input:
%       H: linear model
%       z: observed data
%    meas: measurement model
%    proc: process model, can be 'none'
%  lambda: tradeoff parameter between process and measurement


maxItNonlin = 100;

explicit = ~(isa(H,'function_handle'));

if(explicit)
    params.relTol = 0;  % solve each subproblem to completion
else
    params.relTol = 0.1; % solve each subproblem approximately.
end


REVISION = '$Rev: 2 $';
DATE     = '$Date: 2013-07-01 17:41:32 -0700 (Mon, 01 Jul 2013) $';
REVISION = REVISION(6:end-1);
DATE     = DATE(35:50);

t_start = tic;

% general algorithm parameteres
if(~isfield(params, 'relOpt'))
   params.relOpt = 0; 
end
if(~isfield(params, 'optTol'))
   params.optTol = 1e-7; 
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

% control for restricting conjugate domain
if(~isfield(params, 'uConstraints'))
    params.uConstraints = 0;
end

% control for using predictor-corrector
if(~isfield(params, 'mehrotra'))
    params.mehrotra = 0;
end



if(explicit)
    m = size(H, 1);
    par.m = m;
    n = size(H, 2);
else  % if nonlinear, m must be in params. 
    m = params.m;
    par.m = m;
    n = params.n;
end
    
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
    par_proc.size = n;
    par_proc.mMult = params.proc_mMult;
    par_proc.lambda = params.proc_lambda;
    par_proc.kappa = params.proc_kappa;
    par_proc.tau = params.proc_tau;
    par_proc.scale = params.proc_scale;
    par_proc.eps = params.proc_eps;
    
    [Mw Cw cw bw Bw pFun] = loadPenalty(pLin, k, processPLQ, par_proc);
end

% Load dictionary
if(pFlag && params.dictionary)
    par_proc.mMult = params.dict_mMult;

    par_proc.lambda = params.proc_lambda;
    par_proc.kappa = params.proc_kappa;
    par_proc.tau = params.proc_tau;
    par_proc.scale = params.proc_scale;
    par_proc.eps = params.proc_eps;
    [Md Cd cd bd Bd dFun] = loadPenalty(speye(n), params.dict_vec, 'l2', par_proc);
end


% Define measurement PLQ

par_meas.size = m;
par_meas.mMult = params.meas_mMult;
par_meas.lambda = params.meas_lambda;
par_meas.kappa = params.meas_kappa;
par_meas.tau = params.meas_tau;
par_meas.eps = params.meas_eps;
par_meas.scale = params.meas_scale;


if(explicit)

    [Mv Cv cv bv Bv mFun] = loadPenalty(H, z, measurePLQ, par_meas);
    
    % define objective function
    if(pFlag && params.dictionary)
        params.objFun = @(x) mFun(z - H*x) + pFun(x) + dFun(x + params.dict_vec);
    elseif(pFlag)
        params.objFun = @(x) mFun(z - H*x) + pFun(x);
    else
        params.objFun = @(x) mFun(z - H*x);
    end
    
    
    %%%%%%
    
    K = size(Bv, 1);
    params.pFlag = pFlag;
    params.m = m;
    params.n = n;
    if(pFlag)
        if(params.dictionary)
            [bw, Bw, cw, Cw, Mw] = addPLQFull(bd, Bd, cd, Cd, Md, bw, Bw, cw, Cw, Mw);
        end
            
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
    
    
    if(~params.silent)
        fprintf('\n');
        fprintf(' %s\n',repmat('=',1,80));
        fprintf(' IPsolve  v.%s (%s)\n', REVISION, DATE);
        fprintf(' %s\n',repmat('=',1,80));
        fprintf(' %-22s: %8i %4s'   ,'No. rows'          ,m                 ,'');
        fprintf(' %-22s: %8i\n'     ,'No. columns'       ,n                    );
        fprintf(' %-22s: %8.2e %4s' ,'Optimality tol'    , params.optTol           ,'');
        fprintf(' %-22s: %8.2e\n'   ,'Penalty(b)'        , mFun(z)               );
        fprintf(' %-22s: %8s %4s'   ,'Penalty'  , processPLQ, '    ');
        fprintf(' %-22s: %s\n'     ,'Regularizer'       , measurePLQ);
        fprintf(' %s\n',repmat('=',1,80));
        fprintf('\n');
        
    end
    
    
    params.mu = 0;
    
    Fin = kktSystem(b, Bm, c, C, Mv, sIn, qIn, uIn,  rIn, wIn, yIn,params);
    
    [yOut, uOut, qOut, sOut, rOut, wOut, info] = ipSolver(b, Bm, c, C, Mv, sIn, qIn, uIn, rIn, wIn, yIn, params);
    
    Fout = kktSystem(b, Bm, c, C, Mv, sOut, qOut, uOut, rOut, wOut, yOut, params);
    
    ok = norm(Fout) < 1e-6;
    normFout = norm(Fout); % Added by NRK
    
    if(~params.silent)
        fprintf('KKT In, %5.3f, KKT Final, %5.3f, mu, %f, itr %d\n', norm(Fin), norm(Fout), info.muOut, info.itr);
        fprintf('Obj In, %5.3f, Obj Final, %5.3f \n', params.objFun(yIn), params.objFun(yOut));
        
        
        toc(t_start);
    end
else
    % initialize y 
    y   = zeros(n, 1);
 %   y = randn(n,1);
    converged = 0;
    fprintf('\n');
    fprintf(' %s\n',repmat('=',1,80));
    fprintf(' IPsolve  v.%s (%s)\n', REVISION, DATE);
    fprintf(' %s\n',repmat('=',1,80));
    fprintf(' %-22s: %8i %4s'   ,'No. rows'          ,m                 ,'');
    fprintf(' %-22s: %8i\n'     ,'No. columns'       ,n                    );
    fprintf(' %-22s: %8.2e %4s' ,'Optimality tol'    , params.optTol           ,'');
    fprintf(' %\n-22s: %8s %4s'   ,'Penalty'  , processPLQ, '    ');
    fprintf(' %-22s: %s\n'     ,'Regularizer'       , measurePLQ);
    fprintf(' %s\n',repmat('=',1,80));
    fprintf('\n');

    
    params.silent = 1;
    logB = ' %5i  %13.7f  %13.7f  %13i %13i';
    logH = ' %5s  %13s  %13s  %13s %13s \n';
    fprintf(logH,'Iter','Objective','dirDer','LS-iter','inner');
    fprintf('\n');

    
    
    
    itr = 0;
    tic
    while(~converged)&&itr <= maxItNonlin
        % evaluate z and H
        [Hy Hex] = H(y); 
        zex = z - Hy;
        [Mv Cv cv bv Bv mFun] = loadPenalty(Hex, zex, measurePLQ, par_meas);
        % define objective function
        
        

        K = size(Bv, 1);
        params.pFlag = pFlag;
        params.m = m;
        params.n = n;
        if(pFlag)
            [Mw Cw cw bw Bw pFun] = loadPenalty(pLin, y, processPLQ, par_proc);
            [b, c, C] = addPLQ(bv, cv, Cv, bw, cw, Cw);
            Bm = Bv;
            params.B2 = Bw;
            params.M2 = Mw;
            K = K + size(Bw,1);
        else
            b = bv; Bm = Bv; c = cv; C = Cv; M = Mv;
        end
        C = C';
        
        
        if(pFlag)
            params.objFun = @(x) mFun(H(x) - z) + pFun(x);
            params.objLin = @(x) mFun(Hex*(x) - zex)+pFun(x+y);
        else
            params.objFun = @(x) mFun(H(x) - z);
            params.objLin = @(x) mFun(Hex*x - zex);
            
        end

        
        L = size(C, 2);
        
        sIn = 100*ones(L, 1);
        qIn = 100*ones(L, 1);
        uIn = zeros(K, 1) + 0.01;
        yIn = zeros(n,1);
        
        if(params.constraints)
            P = size(params.A, 2);
            rIn = 10*ones(P, 1);
            wIn = 10*ones(P, 1);
        else
            rIn = [];
            wIn = [];
        end
    
        obj_cur = params.objFun(y);
        
        
        
        % debugging lines
       % obj_cur_affine = params.objLin(0*y);
       % fprintf('obj_cur: %5.4f, obj_cur_affine: %5.4f\n', obj_cur, obj_cur_affine); 
       % fprintf('current: %5.4f\n', obj_cur);
        [yOut, uOut, qOut, sOut, rOut, wOut, info] = ipSolver(b, Bm, c, C, Mv, sIn, qIn, uIn, rIn, wIn, yIn, params);
        
        params.mu = 0;
        F = kktSystem(b, Bm, c, C, Mv, sOut, qOut, uOut, rOut, wOut, yOut, params);

        
        % line search
%        dirDer = params.objLin(yOut) - params.objFun(yOut);
        dirDer = params.objLin(yOut) - obj_cur;
   %     converged = dirDer > -params.optTol;
        converged = dirDer > -params.optTol*1e2; %|| norm(F, inf) < 1e-4;
        
% 
%         if(converged)
%             y_new = yOut;
% %             if(abs(dirDer) < params.optTol*1e2)
% %                 y_new = yOut;
% %                 obj_lambda = params.objFun(y_new);
% %             else
% %                 y_new = yIn;
% %                 obj_lambda = params.objFun(y_new);
% %             end
% %             fprintf(logB, itr, obj_lambda, dirDer, 0, info.itr);
% %             fprintf('\n');
%             break;
%         end
        
        c = 0.001;
        gamma = 0.5;
        lambda = 2.;
        done = false;
        max_search_itr = 50;
        search_itr = 0;
        while(~done)&&(search_itr < max_search_itr)
            search_itr = search_itr + 1;
            lambda = lambda * gamma;
            
            y_new = y + lambda * (yOut);
            obj_lambda = params.objFun(y_new);
            done = ((obj_lambda - obj_cur) <= c*lambda*dirDer);
        %    fprintf('S_itr: %s, Diff: %5.4f, dirDer: %5.4f\n', obj_lambda - obj_cur, dirDer);
        end
        
        if(search_itr == max_search_itr)
            
            fprintf('Norm of F: %5.4f\n', norm(F, inf));
            break
%             if(converged)
%                 break
%             else
%                 error('line search did not converge');
%             end
        end
        
        fprintf(logB, itr, obj_lambda, dirDer, search_itr, info.itr);
        fprintf('\n');
        
        itr = itr + 1;
        
        y = y_new;        
    end
    yOut = y_new;
    toc
end

end