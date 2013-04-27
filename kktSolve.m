% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: GNU General Public License Version 2
% -----------------------------------------------------------------------------

function [ds, dq, du, dy] = kktSolve(b, Bm, c, C, M, s, q, u, y, params)

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
    Tm = T(1:m, 1:m);
    Tn = T(m+1:end, m+1:end);
    Bn = params.B2;
end


% form residual vectors
r1      = -s - C'*u + c;
r2      = mu + (C'*u - c).*q;
if(pFlag)
    r3      = -([Bm*y; Bn*y] - M*u - C*q + b) + C*(r2./s);
else
    r3      = -(Bm*y - M*u - C*q + b) + C*(r2./s);
end

utr = u - T\r3;
if(pFlag)
    r4      = -Bm'*utr(1:m) - Bn'*utr(m+1:end);
else
    r4      = -Bm'*utr;
end


% compute dy
if pFlag && n >= m
    BTB = Bn'*(Tn\Bn); % large sparse matrix
    TBAB = Tm + Bm*(BTB\Bm'); % small dense matrix
    Air4 = BTB\r4;
    dy = Air4 - BTB\((Bm'*(TBAB\(Bm*Air4))));
else
    if(pFlag)
        Omega   =  Bn'*(Tn\Bn)+ Bm'*(Tm\Bm);
    else
        Omega   = Bm'*(T\Bm);
    end   
    dy      = Omega\r4;
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