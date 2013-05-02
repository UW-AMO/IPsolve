% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
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

pCon = params.constraints;

r1 = s + C'*u - c;
r2 = q.*s - mu;
if(pFlag)
    r3 = [Bm*y; Bn*y]- M*u - C*q + b;
else
    r3 = Bm*y - M*u - C*q + b;
end


if(pCon)
    Aw = params.A*w;
    r4 = r + params.A'*y - params.a;
    r5 = w.*r - mu;
else
    Aw = 0;
    r4 = [];
    r5 = [];
end

if(pFlag)
     r6 = Bm'*u(1:m) + Bn'*u(m+1:end) + Aw;
else
     r6 = Bm'*u + Aw;
end


F = [r1;r2;r3;r4;r5;r6];

% don't forget negative sign


end