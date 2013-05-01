% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: GNU General Public License Version 2
% -----------------------------------------------------------------------------

function [yOut, uOut, qOut, sOut, rOut, wOut, info] = ipSolver(b, Bm, c, C, M, s, q, u, r, w, y, params)

if(~isfield(params, 'constraints'))
    params.constraints = 0;
end


converge = 0;
max_itr = 100;
gamma   = .01;
epsilon = 1e-6;


epsComp = 1e-6;
epsF = 1e-6;
itr = 0;

%initialize mu
params.mu = 100;

while ( ~ converge ) && (itr < max_itr)
    
    
    itr = itr + 1;
    [F] = kktSystem(b, Bm, c, C, M, s, q, u, r, w, y, params);
    [ds, dq, du, dr, dw, dy] =  kktSolve(b, Bm, c, C, M, s, q, u, r, w, y, params);

    if(params.constraints)
        ratio      = [ ds ; dq; dr ; dw ] ./ [s ; q ;  r ; w ];
    else
        ratio      = [ ds ; dq] ./ [s ; q ];        
    end
    
    
    ratioMax = max(max( - ratio ));
    
    if (ratioMax <=0)
        lambda = 1;
        
    else
        rNeg = -1./ratio(ratio < 0);
        %min(min(ratio))
        maxNeg = min(min(rNeg));
        lambda = .99*min(maxNeg, 1);
    end
    
    
    % line search
    %
    ok        = 0;
    kount     = 0;
    max_kount = 25;
    beta = 0.5;
    lambda = lambda/beta;
    while (~ok) && (kount < max_kount)
        kount  = kount + 1;
        
        
        lambda = lambda *beta;
        % step of size lambda
        s_new = s + lambda * ds;
        q_new = q + lambda * dq;
        u_new = u + lambda * du;
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
    
    if(~params.silent)
        fprintf('Iter: %d, norm(F): %f, mu: %f\n', itr, G_new, params.mu);
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
    
    
    
    G1 = sum(q.*s);
    converge = (G1 < epsComp) || (G_new < epsF);
    
    % every third step is a corrector
    if ( mod(itr, 3) ~= 1 )
        compMuFrac = G1/(2*length(s));
        muNew = .1*compMuFrac;
        params.mu = muNew;
    end
end
end