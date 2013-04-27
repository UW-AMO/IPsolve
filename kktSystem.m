% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: GNU General Public License Version 2
% -----------------------------------------------------------------------------

function [F] = kktSystem(b, Bm, c, C, M, s, q, u, r, w, y, params)

% Let's see who's in where
% mu                   :   scalar
% dy, y                :   Nx1, dimension of parameter space
% u, du, b             :   Kx1, dimension of U
% c, q, s, dq, ds      :   Lx1, dimension of PLQ specification
% a, r, w, dr, dw      :   Px1, dimension of constraint (A) specification
% B                    :   KxN, K dimension of U
% C                    :   KxL, C' acts on u and takes it to dimension L
% A                    :   NxP, A' acts on y and takes it to a
% M                    :   KxK, all in U space

pFlag = params.pFlag; % is there a process term?



if(pFlag)
   Bn = params.B2; 
end


mu = params.mu; % IP relaxation
m = params.m; 


r1 = s + C'*u - c;
r2 = q.*s - mu;
if(pFlag)
    r3 = [Bm*y; Bn*y]- M*u - C*q + b;
    r4 = Bm'*u(1:m) + Bn'*u(m+1:end);
else
    r3 = Bm*y - M*u - C*q + b;
    r4 = Bm'*u;
end

F = [r1;r2;r3;r4];

% don't forget negative sign


end