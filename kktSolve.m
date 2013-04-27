% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: GNU General Public License Version 2
% -----------------------------------------------------------------------------

function [ds, dq, du, dr, dw, dy] = kktSolve(b, Bm, c, C, M, s, q, u, r, w, y, params)

pFlag = params.pFlag;
pCon = params.constraints;

mu = params.mu;
n = params.n;
m = params.m;

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
end


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


if(pCon)
    A = params.A;
    a = params.a;
    r4      = -(r + A'*y - a);
    r5      = mu + (A'*y - a).*w;
    Awr5r = A*(w + (r5./r));
    SpOmegaMod =  A*WR*A';
else
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
if pFlag && n >= m
    BTB = Bn'*(Tn\Bn) + SpOmegaMod; % large sparse matrix
    TBAB = Tm + Bm*(BTB\Bm'); % small dense matrix
    Air4 = BTB\r6;
    dy = Air4 - BTB\((Bm'*(TBAB\(Bm*Air4))));
else
    if(pFlag)
        Omega   =  Bn'*(Tn\Bn)+ Bm'*(Tm\Bm) + SpOmegaMod;
    else
        Omega   = Bm'*(T\Bm) + SpOmegaMod;
    end   
    dy      = Omega\r6;
end

% compute dw and dr
if(pCon)
    dw      = (r5 + W*params.A'*dy)./r;
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