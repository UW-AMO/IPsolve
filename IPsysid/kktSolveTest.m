function [ok] = kktSolveTest()

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

% generate some random data
N = 2;
K = 3;
L = 4; 
P = 5;

mu  = rand(1);
y   = randn(N,1);
u   = randn(K,1); b   = randn(K,1);
c   = randn(L,1); q   = rand(L,1);      s   = rand(L,1);
a   = randn(P,1); r   = rand(P,1);      w   = rand(P,1);
B   = randn(K,N); C   = randn(K,L);     A   = randn(N,P); 
cM  = randn(K);   M   = cM'*cM;   % positive semidefinite matrix.


% Can test case without constraints. 
%A = 0;

% solve for the step
[ds, dq, du, dr, dw, dy] = kktSolve(a, A, b, B, c, C, M, s, q, u, r, w, y, mu);

answer = [ds;dq;du;dr;dw;dy];


% construct derivative system explicitly
if(~all(A))
FS = [eye(L)     zeros(L)     C'       zeros(L,P) zeros(L,P) zeros(L,N);
      diag(q)    diag(s)    zeros(L,K) zeros(L,P) zeros(L,P) zeros(L,N);
      zeros(K,L)   -C         -M       zeros(K,P) zeros(K,P)   B       ;
      zeros(P,L) zeros(P,L) zeros(P,K)   eye(P)   zeros(P)     A'      ;
      zeros(P,L) zeros(P,L) zeros(P,K)   diag(w)  diag(r)    zeros(P,N);
      zeros(N,L) zeros(N,L)     B'     zeros(N,P)    A       zeros(N)] ;
else
FS = [eye(L)     zeros(L)     C'        zeros(L,N);
      diag(q)    diag(s)    zeros(L,K)  zeros(L,N);
      zeros(K,L)   -C         -M          B       ;
      zeros(N,L) zeros(N,L)     B'      zeros(N)] ;    
end

% construct relaxed KKT conditions automatically
rhs = kktSystem(a, A, b, B, c, C, M, s, q, u, r, w, y, mu);

ok = norm(FS*answer + rhs) < 1e-6;



end