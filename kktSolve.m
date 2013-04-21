% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: GNU General Public License Version 2
% -----------------------------------------------------------------------------

function [ds, dq, du, dy] = kktSolve(b, B, c, C, M, s, q, u, y, params)

pFlag = params.pFlag;
mu = params.mu;
n = params.n;
m = params.m;

% defining auxiliary terms
D = speye(length(q));
D = spdiags(q./s, 0, D);
Q = speye(length(q));
Q = spdiags(q, 0, Q);

T       = M + C*D*C';
% if two pieces, exploit structure
if(pFlag)
    Tn = T(1:n, 1:n);
    Tm = T(n+1:end, n+1:end);
    Bm = params.B2;
end


% form residual vectors
r1      = -s - C'*u + c;
r2      = mu + (C'*u - c).*q;
if(pFlag)
    r3      = -([B*y; params.B2*y] - M*u - C*q + b) + C*(r2./s);
else
    r3      = -(B*y - M*u - C*q + b) + C*(r2./s);
end

utr = u - T\r3;
if(pFlag)
    r4      = -B'*utr(1:n) - Bm'*utr(n+1:end);
else
    r4      = -B'*utr;
end


% compute dy
if pFlag && n >= m
    BTB = B'*(B\Tn); % large sparse matrix
    TBAB = Tm + Bm*BTB*Bm'; % small dense matrix
    Air4 = BTB*r4;
    dy = Air4 - BTB*(Bm'*(TBAB\(Bm*Air4)));
else
    if(pFlag)
        Omega   =  B'*(Tn\B)+ Bm'*(Tm\Bm);
    else
        Omega   = B'*(T\B);
    end   
    dy      = Omega\r4;
end

% compute du
if(pFlag)
    du      = T\(-r3 + [B*dy; params.B2*dy]);
else
    du      = T\(-r3 + B*dy);
end

%compute dq and ds
dq      = (r2 + Q*C'*du)./s;
ds      = r1 - C'*du;


end