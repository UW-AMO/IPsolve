% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [ds, dq, du, dr, dw, dy] = kktSolve(b, Bm, c, C, Mfun, s, q, u, r, w, y, params)

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
    
MM = [MMm, 0*speye(size(MMm,1), size(MMn,2));
    0*speye(size(MMn,1), size(MMm,2)) MMn];



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
if pFlag && n >= m && pSparse
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