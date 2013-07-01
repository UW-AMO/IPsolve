% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [yOut, uOut, qOut, sOut, rOut, wOut, info] = ipSolver(b, Bm, c, C, M, s, q, u, r, w, y, params)

converge = 0;
max_itr = 100;
gamma   = .01;
epsilon = 1e-6;


epsComp = params.optTol;
epsF = params.optTol;
epsMu = params.optTol;
itr = 0;

%initialize mu
params.mu = 100;


if(~params.silent)
    logB = ' %5i  %13.7f  %13.7f  %13.7f';
    logH = ' %5s  %13s  %13s  %13s \n';
    fprintf(logH,'Iter','Objective','KKT Norm','mu');
    fprintf('\n');
end
%printf(logB,undist(iter),undist(rNorm),undist(rErr),undist(rError1),undist(gNorm),log10(undist(stepG)));

    
while ( ~ converge ) && (itr < max_itr)
    
    
    [F] = kktSystem(b, Bm, c, C, M, s, q, u, r, w, y, params);
    
    if(~params.silent)
%        fprintf('%d \t %5.3f\t %5.4f\t\t %5.4f\n', itr, params.objFun(y), norm(F, inf), params.mu);
        fprintf(logB, itr, params.objFun(y), norm(F, inf), params.mu);
        fprintf('\n');
    end

    itr = itr + 1;

    
    params.useChol = 0;
    [ds, dq, du, dr, dw, dy, params] =  kktSolveNew(b, Bm, c, C, M, s, q, u, r, w, y, params);
%    [ds, dq, du, dr, dw, dy] =  kktSolve(b, Bm, c, C, M, s, q, u, r, w, y, params);
    if(any(isnan([ds; dq; du; dr; dw; dy])))
        error('Nans in IPsolve');
    end
    
   if(any(~isreal([ds; dq; du; dr; dw; dy])))
       fprintf('iter = %d\n', itr);
        error('complex values in IPsolve');
    end
   
    
    if(params.constraints)
        ratio      = [ ds ; dq; dr ; dw ] ./ [s ; q ;  r ; w ];
    else
        ratio      = [ ds ; dq] ./ [s ; q ];
    end
    
    if(params.uConstraints)
       ustepsPos = (params.uMax - u(du>0))./(du(du>0));
       ustepsNeg = (params.uMin - u(du<0))./(du(du<0));
       if(isempty(ustepsPos) && isempty(ustepsNeg))
           ustepsMax = 1;
       else 
           if(isempty(ustepsPos))
               ustepsMax = min(ustepsNeg);
           elseif isempty(ustepsNeg)
               ustepsMax = min(ustepsPos);
           else
               ustepsMax = min(min(ustepsPos), min(ustepsNeg));
           end
       end
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

    if(params.uConstraints)
        lambda = min(lambda, 0.99*ustepsMax);
    end
    
    
    if(lambda <0)
        error('negative lambda');
    end
    % line search
    %
    ok        = 0;
    kount     = 0;
    max_kount = 10;
    beta = 0.5;
    lambda = lambda/beta;
    while (~ok) && (kount < max_kount)
        kount  = kount + 1;
        
        
        lambda = lambda *beta;
        % step of size lambda
        s_new = s + lambda * ds;
        q_new = q + lambda * dq;
        u_new = u + lambda * du;
        if(params.uConstraints)
            if(any(u_new > params.uMax))
                error('u_new exceeds max constraints\n');
            elseif (any(u_new < params.uMin))
                error('u_new exceeds min constraints\n');
            end
        end
        
        y_new = y + lambda * dy;
        if(params.constraints)
            r_new = r + lambda * dr;
            w_new = w + lambda * dw;
        else
            r_new = [];
            w_new = [];
        end
        
        
        
        %
        
        
        % check for feasibility
        if min(min(s_new)) <= 0 || min(min(q_new)) <=0
            error('ipSolver: program error, negative entries');
        end
        
        %        [F] = kktSystem(b, Bm, c, C, M, s_new, q_new, u_new, r_new, w_new, y_new, params);
        [F_new] = kktSystem(b, Bm, c, C, M, s_new, q_new, u_new, r_new, w_new, y_new, params);
        
        
        
        G     = max(abs(F));
        G_new = max(abs(F_new));
        
        ok   = (G_new <= (1 - gamma *lambda) * G);
    end
    
    % if using mehrotra extension 
    if(params.mehrotra) %&&  mod(itr, 3) == 1 
%            muOld = params.mu;
            params.mu = params.mu*(1-lambda);
            if(params.mu <0)
                error('negative mu passed through mehrotra');
            end
            
            params.useChol = 0;
            [ds, dq, du, dr, dw, dy, params] =  kktSolveNew(b, Bm, c, C, M, s_new, q_new, u_new, r_new, w_new, y_new, params);
            
            %params.useChol = 1;
            %[ds, dq, du, dr, dw, dy, params] =  kktSolveNew(b, Bm, c, C, M, s, q, u, r, w, y_new, params);

            if(any(isnan([ds; dq; du; dr; dw; dy])))
                error('Nans in IPsolve');
            end
            
            if(any(~isreal([ds; dq; du; dr; dw; dy])))
                fprintf('iter = %d\n', itr);
                error('complex values in IPsolve');
            end
            
            
            if(params.constraints)
                ratio      = [ ds ; dq; dr ; dw ] ./ [s_new ; q_new ;  r_new ; w_new ];
            else
                ratio      = [ ds ; dq] ./ [s_new ; q_new ];
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
            
            if(params.uConstraints)
                lambda = min(lambda, 0.99*ustepsMax);
            end
            

            
            s_new = s_new + lambda*ds;
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
            
            
            if min(min(s_new)) <= 0 || min(min(q_new)) <=0
                fprintf('min of s_new: %5.3f, min of q_new: %5.3f\n', min(min(s_new)), min(min(q_new)));
                error('ipSolver: program error, negative entries in mehrotra part');
            end

            
    end
    
    
    if ~ok
        df = max(F - F_new);
        if(df <= epsilon)
            return
        end
        error('ipSolver: line search failed');
    end
    %F    = F_new;
    %
    s = s_new; sOut = s;
    q = q_new; qOut = q;
    u = u_new; uOut = u;
    y = y_new; yOut = y;
    r = r_new; rOut = r;
    w = w_new; wOut = w;
    info.muOut = params.mu;
    info.itr = itr;
    
    
    if(params.constraints)
        G1 = sum(q.*s) + sum(r.*w);
    else
        G1 = sum(q.*s);
    end
   % converge = (G1 < epsComp) || (G_new < epsF);
    
    % every third step is a corrector
%    if ( mod(itr, 3) ~= 1 )
        if(params.constraints)
            compMuFrac = G1/(2*length(s) + 2*length(r));
        else
            compMuFrac = G1/(2*length(s));
        end
        muNew = 0.1*compMuFrac;
        params.mu = muNew;
 %   end
 converge = (G1 < epsComp) || (G_new < epsF) || params.mu < epsMu;

 if(converge)
     if(~params.silent)
%        fprintf('%d \t %5.3f\t %5.3f\t %f\n', itr, params.objFun(y_new), G_new, params.mu);
       fprintf(logB, itr, params.objFun(y_new), G_new, params.mu);
       fprintf('\n\n');
    end 
 end
 
end

end