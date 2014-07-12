% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [ds, dq, du, dr, dw, dy, params] = kktSolveAction(b, Bm, c, C, Mfun, s, q, u, r, w, y, params)


pCon = params.constraints;
mu = params.mu;
rho = params.rho;
delta = params.delta;

% First, set up RHS
rhs = kktRHS(b, Bm, c, C, Mfun, s, q, u, r, w, y, params);

% Second, set up action
multAction = @(x)kktAction(x, Bm, C, Mfun, s, q, r, w, params);

%trash = multAction(rhs);
% third, solve using pcg
[vecAns] = minres(multAction, rhs, params.tolqual, 10000);

% extract everybody. 
sizeu = size(C,1); % messed this dimension up 
du = -vecAns(1:sizeu);

if(pCon)
    sizew = length(w);
    dw = -vecAns(sizeu+1:sizeu+sizew);
    dr = -r + (mu - r*dw)./w;
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