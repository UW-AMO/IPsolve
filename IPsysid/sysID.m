function [ok] = sysID()

n = 50;
m = 1000;


% generate process information
xTrue = rand(n,1);
tQ = randn(n);
Q = eye(n) + tQ'*tQ;
L = chol(Q)';

% generate measurement information
sigma = .02;
H = randn(m, n);

% generate measurements
z = H*L*xTrue + sigma*randn(m,1);


processPLQ = 'l1';  
processKappa = 1000; % if you choose Huber, and kappa large enough, its least squares

measurePLQ = 'l2';   
measureKappa = 1000; % if you choose Huber, and Kappa large enough, its least squares


constraint = 'none';
boxSize = 0.2;  % if box large enough, constraints not active


% Define process PLQ
% example 1: Huber
switch(processPLQ)
    case 'huber'
        Mw = speye(n);
        Cw    = [speye(n); -speye(n)];
        cw    = processKappa*ones(2*n, 1);
        bw    = zeros(n, 1);
        Bw    = speye(n);
    case 'l1'
        Mw = zeros(n);
        Cw    = [speye(n); -speye(n)];
        cw    = ones(2*n, 1);
        bw    = zeros(n, 1);
        Bw    = speye(n);
    case 'l2'
        Mw = speye(n);
        Cw    = zeros(1,n); % easy to satisfy 0'*u <= 1
        cw    = 1;
        bw    = zeros(n, 1);
        Bw    = speye(n);
    otherwise error('unknown process PLQ');
end
% Define measurement PLQ
% example 1: Huber
switch(measurePLQ)
    case 'huber'
        Mv = speye(m);
        Cv = [speye(m); -speye(m)];
        cv = measureKappa*ones(2*m, 1);
        bv = zeros(m,1);
        Bv = speye(m);
    case 'l1'
        Mv = zeros(m);
        Cv = [speye(m); -speye(m)];
        cv = ones(2*m, 1);
        bv = zeros(m,1);
        Bv = speye(m);
    case 'l2'
        Mv = speye(m);
        Cv    = zeros(1,m); % easy to satisfy 0'*u <= 1
        cv    = 1;
        bv    = zeros(m, 1);
        Bv    = speye(m);
    otherwise error('unknown measurement PLQ');
end
% Define constraints: say our parameters are in a big box
switch(constraint)
    case 'box'
        A = [speye(n); -speye(n)];
        a = boxSize*ones(2*n, 1);
        A = A';
    case 'none'
        A = [];
        a = [];
    otherwise error('Unknown constraint');
end


% Generate data for ipSolver
M = [Mw   zeros(n, m);
    zeros(m, n) Mv ];

B = [Bw; Bv*H*L/sigma];

b = [bw; bv - Bv*z/sigma];

 C = [Cw            zeros(size(Cw,1), size(Cv,2));
     zeros(size(Cv,1), size(Cw,2))           Cv ];


C = C';

c = [cw; cv];

K = size(B, 1);
%N = size(A, 1);
P = size(A, 2);
L = size(C, 2);

sIn = 100*ones(L, 1);
qIn = 100*ones(L, 1);
uIn = zeros(K, 1);
rIn = 100*ones(P, 1);
wIn = 100*ones(P, 1);
yIn   = zeros(n, 1);


Fin = kktSystem(a, A, b, B, c, C, M, sIn, qIn, uIn, rIn, wIn, yIn, 0);

[yOut, wOut, rOut, uOut, qOut, sOut, info] = ipSolver(a, A, b, B, c, C, M, sIn, qIn, uIn, rIn, wIn, yIn);
Fout = kktSystem(a, A, b, B, c, C, M, sOut, qOut, uOut, rOut, wOut, yOut, 0);



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




% construct relaxed KKT conditions automatically

ok = norm(Fout) < 1e-6;

disp(sprintf('In, %f, Final, %f, mu, %f, itr %d\n', norm(Fin), norm(Fout), info.muOut, info.itr));

disp(sprintf('Error, %f', norm(yOut - xTrue)));
disp(sprintf('Infnorm, %f', norm(yOut, inf)));



end