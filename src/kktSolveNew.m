% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [ds, dq, du, dr, dw, dy, params] = kktSolveNew(b, Bm, c, C, Mfun, s, q, u, r, w, y, params)


inexact = params.inexact;

delta = params.delta;


%useChol = params.useChol;
pSparse = params.pSparse;
pFlag = params.pFlag;
pCon = params.constraints;
funM = isa(Mfun, 'function_handle');

mu = params.mu;
n = params.n;
%m = params.m;
m = size(Bm, 1);

% defining auxiliary terms
D = speye(length(q));
D = spdiags(q./s, 0, D);
Q = speye(length(q));
Q = spdiags(q, 0, Q);

if(pCon)
    WR = speye(length(w));
    WR = spdiags(w./r, 0, WR);
    W =  speye(length(w));
    W = spdiags(w, 0, W);
else
    WR = 0;
end

if(pFlag)
    MMn = params.M2;
    um = u(1:m);
    un = u(m+1:end);
else
    um = u;
end

if(funM)
    [Mum, MMm] = Mfun(um);
else
    MMm = Mfun;
    Mum = MMm*um;
end

if(pFlag)
    Mu = [Mum; MMn*un];
else
    Mu = Mum;
end

if(pFlag)
    MM = [MMm + params.rho*speye(size(MMm)), sparse(size(MMm,1), size(MMn,2));
        sparse(size(MMn,1), size(MMm,2)) MMn + params.rho*speye(size(MMn))];
else
    MM = MMm;
end



T       = MM + C*D*C';
% if two pieces, exploit structure
if(pFlag)
    Tm = T(1:m, 1:m);
    Tn = T(m+1:end, m+1:end);
    Bn = params.B2;
end


% form residual vectors
r1      = -s - C'*u + c;
r2      = mu + (C'*u - c).*q;
if(pFlag)
    r3      = -([Bm*y; Bn*y] - Mu - C*q + b) + C*(r2./s);
else
    r3      = -(Bm*y - Mu - C*q + b) + C*(r2./s);
end


if(pCon)
    A = params.A;
    a = params.a;
    r4      = -(r + A'*y - a);
    r5      = mu + (A'*y - a).*w;
    Awr5r = A*(w + (r5./r));
%    SpOmegaMod =  A*WR*A';
else
    A = 0;
    Awr5r = 0;
    SpOmegaMod = 0*speye(n);
end


utr = u - T\r3;
if(pFlag)
    r6      = -Bm'*utr(1:m) - Bn'*utr(m+1:end) - Awr5r ;
else
    r6      = -Bm'*utr - Awr5r;
end


% compute dy
if pFlag && n > m && pSparse &&~inexact
    BTB = Bn'*(Tn\Bn) + SpOmegaMod; % large sparse matrix
    
    Air4 = BTB\r6;
    r = Bm*Air4;
    
    % indirect (working) version
    if(inexact)
        
        TBAB = @(x) Tm*x + Bm*(BTB\(Bm'*x)) + x*params.delta;
        
        %maxElt = max(diag(BTB));
        %precon = diag(max(maxElt*ones(m,1), diag(Tm)));
        [u, FLAG,RELRES,ITER] = pcg(TBAB, r, params.tolqual, 10000);
        params.info.pcgIter = [params.info.pcgIter ITER];
        
        % direct working version
    else
        TBAB = Tm + Bm*(BTB\Bm');
        u = TBAB\r;
    end
    
    dy = Air4 - BTB\(Bm'*u);
    
    
else
    if(pFlag)
        if(inexact)
            if(pCon)
                BTB = Bn'*(Tn\Bn) + A*WR*A' + delta*speye(size(Bn,2));
            else
                BTB = Bn'*(Tn\Bn) + delta*speye(size(Bn,2));
            end
            Omega   =  @(x) BTB*x + Bm'*(Tm\(Bm*x)) + params.delta*x;
 %           dTn = diag(Tn);
%            elt = max(max(dTn(1:m), diag(Tm)));
%            eltTwo = max(max(diag(Tn)), max(diag(Tm)));
%            precon = (1/eltTwo)*(Bn)'*Bn;
            
%            mat1 = (Tm + (params.Anorm^2/delta)*speye(size(Tm)));
 %           mat2 = Tn + 1*speye(size(Tn)); 
%            mat = blkdiag((Tm + (params.Anorm^2/delta)*speye(size(Tm))), Tn + 1*speye(size(Tn)));
            
%            mat3 = BTB + speye(size(BTB))/delta;

%            preCon = @(x) mat3\x - Bm'*(mat1\(Bm*x))/delta^2- Bn'*(mat2\(Bn*x))/delta^2;
%            preCon = @(x) x/delta - [Bm' Bn']*(mat\([Bm*x; Bn*x]))/delta^2;


            
%            preCon = @(x) x/delta - Bm'*((Tm + (params.Anorm^2/delta)*speye(size(Tm)))\(Bm*x))/delta^2;
%            [dy] = pcg(Omega, r6, params.tolqual, 10000, preCon);

            [dy, FLAG,RELRES,ITER] = pcg(Omega, r6, params.tolqual, 10000);
            params.info.pcgIter = [params.info.pcgIter ITER];

            
            

%            [dy] = minres(Omega, r6, params.tolqual, 10000);
            
            
%            preCon = @(x) x/delta - Bm'*(Tm\(Bm*x));

            
        else
            Omega   =  Bn'*(Tn\Bn)+ Bm'*(Tm\Bm) + A*WR*A'+delta*speye(size(Bm,2));
            OmegaChol = chol(Omega);
            dy      = OmegaChol\(OmegaChol'\r6);
        end
    else
        if(inexact)
            if(pCon)
                
                Omega   =  @(x) A*(WR*(A'*x)) + Bm'*(T\(Bm*x)) + delta*x;
            else
                Omega   =  @(x) Bm'*(T\(Bm*x)) + delta*x;
            end

      %      preCon = @(x) x/delta - Bm'*((T + (params.Anorm^2/delta)*speye(size(T)))\(Bm*x))/delta^2;
      %      [dy] = pcg(Omega, r6, params.tolqual, 10000, preCon);
            
            
            [dy, FLAG,RELRES,ITER] = pcg(Omega, r6, params.tolqual, 10000); 
            params.info.pcgIter = [params.info.pcgIter ITER];

            
        else
            if(pCon)
                Omega   = Bm'*(T\Bm) + A*WR*A' + delta*speye(size(Bm,2));
            else
                Omega   = Bm'*(T\Bm) + delta*speye(size(Bm,2));
            end
            %dy = Omega\r6;
            OmegaChol = chol(Omega);
            dy      = OmegaChol\(OmegaChol'\r6);
        end
    end
    
    %dy = Omega\r6;
    %    [dy, ~] = lsqr(Omega, r6, [], 5);
end

% compute dw and dr
if(pCon)
    dw      = (r5 + W*(params.A'*dy))./r;
    dr      = r4 - params.A'*dy;
else
    dw = [];
    dr = [];
end

% compute du
if(pFlag)
    du      = T\(-r3 + [Bm*dy; Bn*dy]);
else
    du      = T\(-r3 + Bm*dy);
end

%compute dq and ds
dq      = (r2 + Q*C'*du)./s;
ds      = r1 - C'*du;

end