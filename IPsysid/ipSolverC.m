function [yOut, wOut, rOut, uOut, qOut, sOut, info] = ipSolver(a, A, b, B, c, C, M, s, q, u, r, w, y)

converge = 0;
max_itr = 100;
gamma   = .01;

epsilon = 1e-6;
itr = 0;

%initialize mu
mu = 100;

while ( ~ converge ) && (itr < max_itr)



    itr = itr + 1;
    F                        = kktSystem(a, A, b, B, c, C, M, s, q, u, r, w, y, mu);
    [ds, dq, du, dr, dw, dy] =  kktSolve(a, A, b, B, c, C, M, s, q, u, r, w, y, mu);


    % Useful debugging tool from legacy code: SHOULD BE 0!!!
    %         expr1 =    norm(pMinus(:).*s(:) + s(:).*dPMinus(:) + pMinus(:).*ds(:) - mu, inf);
    %         expr2 =    norm(pPlus(:).*r(:) + r(:).*dPPlus(:) + pPlus(:).*dr(:) - mu, inf);


    %
    % determine maximum allowable step factor lambda
    %ratio     = [ dr ; ds; dPPlus; dPMinus ] ./ [ r ; s; pPlus; pMinus ];

    if(~all(A))
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
    max_kount = 18;
    beta = 0.5;
    lambda = lambda/beta;
    while (~ok) && (kount < max_kount)
        kount  = kount + 1;


        lambda = lambda *beta;
        % step of size lambda
        s_new = s + lambda * ds;
        q_new = q + lambda * dq;
        u_new = u + lambda * du;
        if(~all(A))
            r_new = r + lambda * dr;
            w_new = w + lambda * dw;
        else
            r_new = [];
            w_new = [];
        end

        y_new = y + lambda * dy;

        %


        % check for feasibility
        if(~all(A))
            if min(min(s_new)) <= 0 || min(min(q_new)) <=0 ||min(min(r_new))<=0 || min(min(w_new)) <= 0
                error('ipSolver: program error, negative entries');
            end
        else
            if min(min(s_new)) <= 0 || min(min(q_new)) <=0
                error('ipSolver: program error, negative entries');
            end
        end


        F_new = kktSystem(a, A, b, B, c, C, M, s_new, q_new, u_new, r_new, w_new, y_new, mu);


        G     = max(abs(F));
        G_new = max(abs(F_new));

        ok   = (G_new <= (1 - gamma *lambda) * G);
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
    r = r_new; rOut = r;
    w = w_new; wOut = w;
    y = y_new; yOut = y;
    info.muOut = mu;
    info.itr = itr;



    %VP = ckbs_L2L1_obj(y, z, g, h, dg, dh, qinv, rinv);
    %Kxnu = ckbs_L2L1_obj(xZero, z, g, h, dg, dh, qinv, rinv);

    G1 = sum(r.*w) + sum(q.*s);
    converge = G1 < epsilon;
    %converge = (G1 < min(Kxnu - VP, epsilon));
    %converge = (G1 < Kxnu - VP);

    % every third step is a corrector
    if ( mod(itr, 3) ~= 1 )
        compMuFrac = G1/(2*length(r) + 2*length(s));
        muNew = .1*compMuFrac;
        mu = muNew;
    end
end
end