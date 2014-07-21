% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [ds, dq, du, dr, dw, dy, params] = kktSolveAction(b, Bm, c, C, Mfun, s, q, u, r, w, y, params)


pFlag = params.pFlag;
pCon = params.constraints;
mu = params.mu;
funM = isa(Mfun, 'function_handle');


% First, set up RHS
rhs = kktRHS(b, Bm, c, C, Mfun, s, q, u, r, w, y, params);

m = size(Bm, 1);

if(pFlag)
%    MMn = params.M2;
    um = u(1:m);
%    un = u(m+1:end);
else
    um = u;
end

if(funM)
    [~, MMm] = Mfun(um);
else
    MMm = Mfun;
%    Mum = MMm*um;
end


% Second, set up action
multAction = @(x)kktAction(x, Bm, C, MMm, s, q, r, w, params);


%mat = kktPrecond(Bm, Mfun, s, q, r, w, params);
%matInv = mat\speye(size(mat));

[vecAns] = minres(multAction, rhs, params.tolqual, 10000);
%[vecAns] = symmlq(multAction, rhs, params.tolqual, 10000,matInv);


% extract everybody. 
sizeu = size(C,1); % messed this dimension up 
du = -vecAns(1:sizeu);

if(pCon)
    sizew = length(w);
    dw = -vecAns(sizeu+1:sizeu+sizew);
    dr = -r + (mu - r.*dw)./w;
else
    sizew = 0;
    dw = [];
    dr = [];
end

sizeq = length(q);
dq = -vecAns(sizeu+sizew+1:sizeu+sizew+sizeq);

ds = -s + (mu - s.*dq)./q;

dy = vecAns(sizeu+sizew+sizeq+1:end);


end