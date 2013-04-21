function [ok] = ipSolverTest()

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


% relationships: 
% 1) B is supposed to be injective, so K >= N. 



% generate some random data
N = 2;
K = 3;
L = 4; 
P = 5;

y   = randn(N,1);
u   = randn(K,1); b   = randn(K,1);
c   = randn(L,1); q   = rand(L,1);      s   = 1 + rand(L,1);
a   = randn(P,1); r   = rand(P,1);      w   = rand(P,1);
B   = randn(K,N) + eye(K,N); C   = randn(K,L) + eye(K,L);     A   = randn(N,P); 
cM  = randn(K) + eye(K);   M   = cM'*cM;   % positive (semi)definite matrix.

A = [];


% construct relaxed KKT conditions automatically
% initial F: 
Fin = kktSystem(a, A, b, B, c, C, M, s, q, u, r, w, y, 0);
[yOut, wOut, rOut, uOut, qOut, sOut,info] = ipSolver(a, A, b, B, c, C, M, s, q, u, r, w, y);

Fout = kktSystem(a, A, b, B, c, C, M, sOut, qOut, uOut, rOut, wOut, yOut, 0);


ok = norm(Fout) < 1e-6;

disp(sprintf('In, %f, Final, %f, mu, %f, itr %d', norm(Fin), norm(Fout), info.muOut, info.itr));

end