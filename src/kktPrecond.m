% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [mat] = kktPrecond(Bm, M, s, q, r, w, params)

% v: vector to take product with 

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




% SASHA: this crazy thing is off for now. 
%funM = isa(M, 'function_handle'); % for now, assume only measurement could be this way


rho = params.rho;
delta = params.delta;

pFlag = params.pFlag; % is there a process term?
pCon = params.constraints; % forgot to do this

if(~pFlag)
    Mf = M;
else
    Mf = [M zeros(size(M,1), size(params.M2,2)); zeros(size(params.M2,1), size(M,2)) params.M2];
end

Mf = Mf + rho*speye(size(Mf));



QS = spdiags(s./q, 0, length(q), length(q));


dI = delta*speye(size(Bm,2));

if(pCon)
    WR = spdiags(r./w, 0, length(r), length(r));
    mat = blkdiag(Mf, WR, QS, dI);
else
    mat = blkdiag(Mf, QS, dI);
end

end