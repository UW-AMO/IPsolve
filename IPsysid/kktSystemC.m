function [F] = kktSystem(a, A, b, B, c, C, M, s, q, u, r, w, y, mu)

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

r1 = s + C'*u - c;
r2 = q.*s - mu;
r3 = B*y - M*u - C*q + b;
if(~all(A))
    r4 = r + A'*y - a;
    r5 = w.*r - mu;
    r6 = B'*u + A*w;
else
    r4 = [];
    r5 = [];
    r6 = B'*u;
end

F = [r1;r2;r3;r4;r5;r6];
% don't forget negative sign


end