function [ x, IMat, s, lam] = sqIP( x, params )

%initial values
s = ones(size(params.d));
lam = ones(size(params.d));
mu = 1;
m = length(params.d);
converged = 0; 
itr = 0; 
maxItr = params.maxItr; 
while(~converged && itr < maxItr)
    itr = itr+1;
    % compute F0
    [r1, r2, r3] = mukkt(s, lam, x, 0, params);
    G = norm([r1; r2; r3], inf);

    [r1, r2, r3] = mukkt(s, lam, x, mu, params);
    
    rhs1 = -r1;
    rhs2 = -r2 - lam.*rhs1;
    rhs3 = -r3 -params.C'*(rhs2./s);

    lamC = sparse(1:m, 1:m, lam)*params.C;
    SinvC = sparse(1:m, 1:m, 1./s)*params.C;
    IMat = params.A'*params.A + lamC'*SinvC;
    
    dx = IMat\rhs3; 
    dlam = (rhs2 + lam.*(params.C*dx))./s; 
    ds = rhs1 - params.C*dx; 
    
    
    ratio      = [ ds ; dlam] ./ [s ; lam ];

    ratioMax = max(max( - ratio ));
    
    
    if (ratioMax <=0)
        alpha = 1;
    else
        rNeg = -1./ratio(ratio < 0);
        %min(min(ratio))
        maxNeg = min(min(rNeg));
        alpha = .9999*min(min(maxNeg),1);
    end
    
    
    % do line search 
    ok        = 0;
    kount     = 0;
    max_kount = 20;
    beta = 0.2;
    alpha = alpha/beta;
    gam = 0.001;
    while (~ok) && (kount < max_kount)
        kount  = kount + 1;
        
        alpha = alpha *beta;
        % step of size alpha
        s_new = s + alpha * ds;
        lam_new = lam + alpha * dlam;
        x_new = x + alpha * dx;
                
        
        % check for feasibility
        if min(min(s_new)) <= 0 || min(min(lam_new)) <=0
            error('spIP: program error, negative entries');
        end
        
        %        [F] = kktSystem(b, Bm, c, C, M, s_new, q_new, u_new, r_new, w_new, y_new, params);
        %[F_new] = kktSystem(b, Bm, c, C, M, s_new, q_new, u_new, r_new, w_new, y_new, params);
        [r1_new, r2_new, r3_new] = mukkt(s_new, lam_new, x_new, 0, params); 
        
        G_new = norm([r1_new; r2_new; r3_new], inf);        
        ok   = (G_new <= (1 - gam *alpha) * G);
        
    end
    converged = (G_new < params.Ftol) ||  (mu < params.muTol);
    s = s_new;
    lam =lam_new;
    x = x_new;

    
    mu = min(0.9*mu, 0.01* sum(s.*lam)/m);
    objVal = 0.5*norm(params.A*x - params.b)^2;
    grNr = norm(params.A'*(params.A*x-params.b));
    %  fprintf('nrmF: %7.2e, mu: %7.2e, step: %7.2e, obj: %7.2e, grNr: %7.2e\n', G, mu, alpha, objVal, grNr);
    % update mu! 
        
end


end

