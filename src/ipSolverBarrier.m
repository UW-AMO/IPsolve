% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [yOut, uOut, qOut, rOut, wOut, info] = ipSolverBarrier(linTerm, b, Bm, c, C, M, q, u, r, w, y, params)

if(~params.inexact)
    params.info.pcgIter = -1;
end

converge = 0;
max_itr = 20;
gamma   = .01;
epsilon = 1e-6;

yIn = y;

relOpt = params.relOpt;
epsComp = params.optTol;
epsF = params.optTol;
epsMu = params.optTol;


% cg interior parameter
params.tolqual = 1e-5;
itr = 0;

%initialize mu
params.mu = 1;


if(~params.silent)
    logB = ' %5i  %13.1e  %13.1e  %13.1e %13i %13.1e %13i ';
    logH = ' %5s  %13s  %13s  %13s %13s %13s %13s\n';
    fprintf(logH,'Iter','Objective','KKT Norm','mu', 'pcgIter', 'inStep', 'numArmijo');
    fprintf('\n');
end
%printf(logB,undist(iter),undist(rNorm),undist(rErr),undist(rError1),undist(gNorm),log10(undist(stepG)));

G_in = 0;    

while ( ~ converge ) && (itr < max_itr)
    
    
    [F] = kktSystemBarrier(linTerm, b, Bm, c, C, M, q, u, r, w, y, params);

    % store norm of initial KKT system
    if(itr == 1)
        G_in = norm(F, inf);
    end
    
    
    % dominique pcg tolerance suggestion
    params.useChol = 0;
    params.tolqual = min(params.tolqual, norm(F, 2)/100);
    params.tolqual = max(params.tolqual, 1e-9);
%    fprintf('params.tolqual is %5.8f\n', params.tolqual);

    
   
    

   
% DEBUG complicated
%    fullMat = [M C Bm; ...
%            C' -diag(s./q) zeros(length(s), size(Bm,2)); ...
%            Bm' zeros(size(Bm,2), length(q)) -0*eye(size(Bm,2))];
%    
%    rhs1 = -Bm*y + M*u +C*q - b;
%    rhs2 = C'*u -c - params.mu./q;
%    rhs3 = Bm'*u;
%    rhsEx = [rhs1; rhs2; rhs3];
% 
%    ansVec = fullMat\rhsEx;
%    fprintf('accuracy of inexact solution: %7.1e\n', norm(ansVec - [-du; -dq; dy]));
% 
%    z = ansVec - [-du; -dq; dy];
%    n1 = length(u);
%    n2 = length(q);
%    norm(z(1:n1))
%    norm(z(n1+1:n1+n2))
%    norm(z(n1+n2+1:end))

% DEBUG simple
%[dsa, dqa, dua, dra, dwa, dya, params] =  kktSolveNew(b, Bm, c, C, M, s, q, u, r, w, y, params);

%   fprintf('ds: %7.1e, dq: %7.1e, du: %7.1e, dr: %7.1e, dw: %7.1e, dy: %7.1e\n', norm(dsa-ds,inf), norm(dqa-dq,inf), norm(dua-du,inf), norm(dra-dr,inf),  norm(dwa-dw,inf), norm(dya-dy,inf));
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KKT Solvers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% One: full solve
%[ds, dq, du, dr, dw, dy] =  kktSolve(b, Bm, c, C, M, s, q, u, r, w, y, params);

% Two: pcg on schur complement:
[dq, du, dr, dw, dy, params] =  kktSolveBarrier(linTerm, b, Bm, c, C, M, q, u, r, w, y, params);

% Three: minres on new system:

%[ds, dq, du, dr, dw, dy, params] =  kktSolveAction(b, Bm, c, C, M, s, q, u, r, w, y, params);

%%

    if(any(isnan([dq; du; dr; dw; dy])))
        error('Nans in IPsolve');
    end
    
   if(any(~isreal([dq; du; dr; dw; dy])))
       fprintf('iter = %d\n', itr);
        error('complex values in IPsolve');
   end
   
   % first try
    d = c - C'*u;
    if(any(d <0))
        error('infeasible d');
    end
    
    dd = -C'*du;
    
    if(params.constraints)
        ratio      = [ dd ; dq; dr ; dw ] ./ [d ; q ;  r ; w ];
    else
        ratio      = [ dd ; dq] ./ [d ; q ];
    end

%     rNeg = -1./ratio(ratio < 0);
%     maxNeg = min(min(rNeg));
%     lambda_in_before = .99*min(min(maxNeg),1);
    
%     if(params.uConstraints)
%        ustepsPos = (params.uMax - u(du>0))./(du(du>0));
%        ustepsNeg = (params.uMin - u(du<0))./(du(du<0));
%        if(isempty(ustepsPos) && isempty(ustepsNeg))
%            ustepsMax = 1;
%        else 
%            if(isempty(ustepsPos))
%                ustepsMax = min(ustepsNeg);
%            elseif isempty(ustepsNeg)
%                ustepsMax = min(ustepsPos);
%            else
%                ustepsMax = min(min(ustepsPos), min(ustepsNeg));
%            end
%        end
%     end
        
    ratioMax = max(max( - ratio ));
    
    
    if (ratioMax <=0)
        lambda = 1;
    else
        rNeg = -1./ratio(ratio < 0);
        %min(min(ratio))
        maxNeg = min(min(rNeg));
        lambda = .99*min(min(maxNeg),1);
    end

%     if(params.uConstraints)
%         lambda = min(lambda, 0.99*ustepsMax);
%     end
    
    

    
    if(lambda <0)
        error('negative lambda');
    end
    % line search
    %
    lambda_in = lambda;
    ok        = 0;
    kount     = 0;
    max_kount = 20;
    beta = 0.2;
    lambda = lambda/beta;
    while (~ok) && (kount < max_kount)
        kount  = kount + 1;
        
        
        lambda = lambda *beta;
        % step of size lambda
%        s_new = s + lambda * ds;
        q_new = q + lambda * dq;
        u_new = u + lambda * du;
%         if(params.uConstraints)
%             if(any(u_new > params.uMax))
%                 error('u_new exceeds max constraints\n');
%             elseif (any(u_new < params.uMin))
%                 error('u_new exceeds min constraints\n');
%             end
%         end
        
        y_new = y + lambda * dy;
        if(params.constraints)
            r_new = r + lambda * dr;
            w_new = w + lambda * dw;
        else
            r_new = [];
            w_new = [];
        end
        
        
        
        %
        

        if min(min(q_new)) <=0
            error('ipSolver: program error, negative entries');
        end

        % check for feasibility
%         if min(min(s_new)) <= 0 || min(min(q_new)) <=0
%             error('ipSolver: program error, negative entries');
%         end
        
        %        [F] = kktSystem(b, Bm, c, C, M, s_new, q_new, u_new, r_new, w_new, y_new, params);
        [F_new] = kktSystemBarrier(linTerm, b, Bm, c, C, M, q_new, u_new, r_new, w_new, y_new, params);
        
        
        %% STOPPED
        
        
        G     = max(abs(F));
        G_new = max(abs(F_new));
        
        ok   = (G_new <= (1 - gamma *lambda) * G);
    end
    
        if(~params.silent)
     %   fprintf(logB, itr, params.objFun(y), norm(F, inf), params.mu, params.info.pcgIter(end), lambda_in, kount);
        fprintf(logB, itr, params.objFun(y), norm(F, inf), params.mu, params.info.pcgIter(end), lambda_in, kount);

        fprintf('\n');
    end

    itr = itr + 1;

    
    
    % SASHA: note tweak for failed line search. 
     if ~ok
         fprintf('Line search failed, returning\n');
         df = max(F - F_new);
         if(df <= epsilon)
            return
         end
        error('ipSolver: line search failed');
    end
    
    % if using mehrotra extension 
    if(params.mehrotra) %&&  mod(itr, 3) == 1 
%            muOld = params.mu;
            params.mu = params.mu*(1-lambda);
            if(params.mu <0)
                error('negative mu passed through mehrotra');
            end
            
            params.useChol = 0;
            [dq, du, dr, dw, dy, params] =  kktSolveBarrier(linTerm, b, Bm, c, C, M, q_new, u_new, r_new, w_new, y_new, params);
            
            if(any(isnan([dq; du; dr; dw; dy])))
                error('Nans in IPsolve');
            end
            
            if(any(~isreal([dq; du; dr; dw; dy])))
                fprintf('iter = %d\n', itr);
                error('complex values in IPsolve');
            end
            
            
            d_new = c - C'*u_new;
            dd = -C'*du;
            
            if(params.constraints)
                ratio      = [ dd ; dq; dr ; dw ] ./ [d_new ; q_new ;  r_new ; w_new ];
            else
                ratio      = [ dd ; dq] ./ [d_new ; q_new ];
            end
            
            ratioMax = max(max( - ratio ));
    
            
            if (ratioMax <=0)
                lambda = 1;
            else
                rNeg = -1./ratio(ratio < 0);
                %min(min(ratio))
                maxNeg = min(min(rNeg));
                lambda = .99*min(min(maxNeg),1);
            end
            
%             if(params.uConstraints)
%                 lambda = min(lambda, 0.99*ustepsMax);
%             end
            

            
            q_new = q_new + lambda*dq;
            u_new = u_new + lambda*du;
            if(params.constraints)
                r_new = r_new + lambda*dr;
                w_new = w_new + lambda*dw;
            else
               r_new = [];
               w_new = [];
            end
            y_new = y_new + lambda*dy;
            %params.mu = muOld;
            
            
            if  min(min(q_new)) <=0
                fprintf('min of q_new: %5.3f\n', min(min(q_new)));
                error('ipSolver: program error, negative entries in mehrotra part');
            end

            
    end
    
    
   
    %F    = F_new;
    %
    q = q_new; qOut = q;
    u = u_new; uOut = u;
    y = y_new; yOut = y;
    r = r_new; rOut = r;
    w = w_new; wOut = w;
    info.muOut = params.mu;
    info.itr = itr;
    

    
    %% SASHA check
    if(params.constraints)
        G1 = sum(q.*d) + sum(r.*w);
    else
        G1 = sum(q.*d);
    end
    
%     if(params.constraints)
%         G1 = sum(q.*s) + sum(r.*w);
%     else
%         G1 = sum(q.*s);
%     end

    
   % converge = (G1 < epsComp) || (G_new < epsF);
    
    % every third step is a corrector
%    if ( mod(itr, 3) ~= 1 )
        if(params.constraints)
            compMuFrac = G1/(2*length(q) + 2*length(r));
        else
            compMuFrac = G1/(2*length(q));
        end
        muNew = 0.1*compMuFrac;
        params.mu = muNew;
 %   end

 if(isfield(params, 'objLin'))
     VP = params.objLin(y);
     Kxnu = params.objLin(yIn);
     converged = (compMuFrac < params.relTol*1e-4*(Kxnu - VP));
%    converged = G_new < 1e-3*G_in;
 else
     converged = 0;
 end
 converge = converged || (G1 < epsComp) || (G_new < epsF) || params.mu < epsMu|| G_new < relOpt*G_in;

	
 
 if(converge)
     if(~params.silent)
%        fprintf('%d \t %5.3f\t %5.3f\t %f\n', itr, params.objFun(y_new), G_new, params.mu);
if(isfield(params, 'objLin'))
    fprintf(logB, itr, params.objLin(y_new), G_new, params.mu);
else
    fprintf(logB, itr, params.objFun(y_new), G_new, params.mu);
end   
fprintf('\n\n');
    end 
 end
 
end

end