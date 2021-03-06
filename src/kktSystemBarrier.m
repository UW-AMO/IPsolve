% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [F] = kktSystemBarrier(linTerm, b, Bm, c, C, M, q, u, r, w, y, params)

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

pFlag = params.pFlag; % is there a process term?

funM = isa(M, 'function_handle'); % for now, assume only measurement could be this way

if(pFlag)
   Bn = params.B2; 
   Mn = params.M2;
end


mu = params.mu; % IP relaxation
%m = params.m; 
m = size(Bm, 1);

pCon = params.constraints;

% new diagonal guy 
d = c - C'*u;

% done with first part
r1 = d.*q -mu;  


if(pFlag)
    um = u(1:m);
    un = u(m+1:end);
else
    um = u;
end

if(funM)
   Mum = M(um); 
else
   Mum = M*um;
end

if(pFlag)
   Mu = [Mum; Mn*un];  
else
    Mu = Mum;
end


% second part remarkably similar
if(pFlag)
    r2 = [Bm*y; Bn*y]- Mu - C*q + b;
else
    r2 = Bm*y - Mu - C*q + b;
end


% third and fourth need tiny changes
if(pCon)
    if(isa(params.A, 'function_handle'))
        Aw = params.A(w,1);
    else
        Aw = params.A*w;
    end
    if(isa(params.A, 'function_handle'))
        Aty = params.A(y, 2);
    else
        Aty = params.A'*y;
    end
    r4 = r + Aty - params.a;
    r3 = w.*r - mu;
else
    Aw = 0;
    r4 = [];
    r3 = [];
end


% fifth part almost identical
if(pFlag)
     r5 = Bm'*um + Bn'*un + Aw + linTerm;
else
     r5 = Bm'*um + Aw + linTerm;
end


F = [r1;r2;r3;r4;r5];



end