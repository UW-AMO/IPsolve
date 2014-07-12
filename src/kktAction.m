% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [prod] = kktAction(v, Bm, C, M, s, q, r, w, params)

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
    n1 = size(M,2);
    v1 = v(1:n1);
else
    n11 = size(Bm,1);
    
    % n12 set incorrectly in past
    n12 = size(C,1) - n11;
    n1 = n11+n12;
    Bn = params.B2;
    Mn = params.M2;
    v11 = v(1:n11);
    v12 = v(n11+1:n11+n12);
    v1 = [v11; v12];
end
n2 = length(r);
v2 = v(n1+1: n1+n2);

n3 = length(s);
v3 = v(n1+n2+1: n1+n2+n3);

%n4 = size(Bm, 2);
v4 = v(n1+n2+n3+1:end);


if(pFlag)
    out1a = M*v11 +rho*v11;
    out1b = Mn*v12+rho*v12;
    out1 = [out1a; out1b];
%   out1 = [M*v11 +rho*v11; Mn*v12+rho*v12];  
   Bv4 = [Bm*v4; Bn*v4];
else
    out1 = M*v1 + rho*v1;
    Bv4 = Bm*v4;
end
out1 = out1 + C*v3 + Bv4;  % finished with out1


if(pCon)
   out2 = (r./w).*v2;
   out2 = out2 + params.A'*v4;
else
    out2 = [];
end                        % finished with out2


% jesus christ - had q./s
out3 = C'*v1- (s./q).*v3;  % finished with out3

if(pFlag)
    out4 = Bm'*v11 + Bn'*v12;
else
    out4 = Bm'*v1;
end

if(pCon)
    out4 = out4 + params.A*v2;
end
out4 = out4 - delta*v4;


prod = [out1; out2; out3; out4];



end