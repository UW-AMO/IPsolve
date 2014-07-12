% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [rhs] = kktRHS(b, Bm, c, C, M, s, q, u, r, w, y, params)

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

% Caution: this will work here, but is not working for kktAction.  
funM = isa(M, 'function_handle'); % for now, assume only measurement could be this way

if(pFlag)
   Bn = params.B2; 
   Mn = params.M2;
end


mu = params.mu; % IP relaxation
%m = params.m; 
m = size(Bm, 1);

pCon = params.constraints;


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



if(pFlag)
    fu = -[Bm*y; Bn*y] + Mu + C*q - b;
else
    fu = -Bm*y + Mu + C*q - b;
end

rhs1 = fu;  % done with first guy. 

if(pCon)
    Aw = params.A*w;
    fw = -r - params.A'*y + params.a;
    fr = -w.*r + mu;
    rhs2 = fw - fr./w;
else
    Aw = 0;
    rhs2 = [];
end         % done with second guy. 


%old
%fq = -s - C'*u + c;
%fs = -q.*s + mu;
%rhs3 = -fq + fs./q;  % done with third guy. 

%new
rhs3 = C'*u -c +mu./q;



if(pFlag)
     fy = Bm'*um + Bn'*un + Aw;
else
     fy = Bm'*um + Aw;
end

rhs4 = fy;


rhs = [rhs1; rhs2; rhs3; rhs4];

% don't forget negative sign


end