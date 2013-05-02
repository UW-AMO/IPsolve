% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

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

params.pFlag = 1;
params.constraints = 1;
constraint = 'box';
boxSize = 0.2;

% Define constraints: say our parameters are in a big box
switch(constraint)
    case 'box'
        A = [speye(n); -speye(n)];
        params.a = boxSize*ones(2*n, 1);
        params.A = A';
    case 'pos'
        A = -speye(n);
        params.a = zeros(n, 1);
        params.A = A';
    case 'none'
        A = [];
        a = [];
    otherwise error('Unknown constraint');
end



yOut = run_example( H, z, 'l2', 'l1', params );





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

switch(constraint)
    case 'box'
       fprintf('Infnorm: %f\n', norm(yOut, inf));
    case 'pos'
       fprintf('maximum negative element: %f\n', norm(pos(-yOut), inf));

    otherwise 
        error('unknown constraint');

end