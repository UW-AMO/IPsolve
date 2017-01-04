% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [F,J] = kktSystemBarrierAffine(linTerm, b, Bm, c, C, M, q, u, r, w, y, zeta, params)

% assume E, e are in params for simplicity
% J is the jacobian of the entire system

% Let's see who's in where
% linTerm              :   linear term in the objective
% mu                   :   scalar
% dy, y                :   Nx1, dimension of parameter space
% u, du, b             :   Kx1, dimension of U
% c, q, s, dq, ds      :   Lx1, dimension of PLQ specification
% a, r, w, dr, dw      :   Px1, dimension of constraint (A) specification
% B                    :   KxN, K dimension of U
% C                    :   KxL, C' acts on u and takes it to dimension L
% A                    :   NxP, A' acts on y and takes it to a
% M                    :   KxK, all in U space

% assume M not a function too 
% assume pFlag = 0;
% assume pCon = 0; (no inequality constraints or regularizers).

mu = params.mu; % IP relaxation

% new diagonal guy
d = c - C'*u;

r1 = d.*q -mu;

td = length(d);
ty = length(y);
te = length(params.e);


tb = length(b);
r2 = Bm*y - M*u - C*q + b;


r3 = Bm'*u + linTerm + params.E'*zeta;




if(params.eqFlag)
    r4 = params.E*y - params.e;
    F = [r1;r2;r3; r4];
 
    if(nargout > 1)
        R1 = [sparse(1:td, 1:td, d), -sparse(1:td, 1:td, q)*C', sparse(td, ty), sparse(td, te)];
        R2 = [-C, -M, Bm, sparse(tb, te)];
        R3 = [sparse(ty, td), Bm', sparse(ty, ty), params.E'];
        R4 = [sparse(te, td), sparse(te, ty), params.E, sparse(te, te)];
        J = [R1; R2; R3;R4];
    end
else
    F = [r1;r2;r3];
    if(nargout > 1)
        R1 = [sparse(1:td, 1:td, d), -sparse(1:td, 1:td, q)*C', sparse(td, ty), sparse(td, te)];
        R2 = [-C, -M, Bm, sparse(tb, te)];
        R3 = [sparse(ty, ty), Bm', sparse(ty, ty), params.E'];
        J = [R1; R2; R3];
    end
end



end