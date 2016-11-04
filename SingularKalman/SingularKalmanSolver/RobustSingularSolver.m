function [soln,  time ] = RobustSingularSolver(A, D, hatw, n, m, N, lambda, othersoln )
%RobustSingluarSolver 
    %This solves the optimization problem given by:
    %         min_z ||Dz||^2 + lambda*||Az - hatw||_1
    %using IPSolve
    %
    %A, D, hatw should be inputed from the SingularKalmanSetup
    %
    %n is the size of the states and m is the size of the measurments and N
    %is the total number of measurments
    %
    %othersoln can be used to specify the output solution to be the state
    %solution, wsolution or vsolution. Enter 'w' or 'v' or 'state for the w or v
    %solution respectively. The defalut is state solution
    %
    %The statesoln is is a n x N+1 matrix whose columns are the estimated
    %states (note it includes the initial state)
    %
    %wsoln and vsoln are the estimated errors given as n x N and m x N
    %matrix respectively with columns of the respective errors(note these
    %do not have initial states)
    %
    %time is the time taken to solve the system
    
    
params.K = D;
[Dr Dc] = size(D);
params.k = zeros(Dr,1);
params.inexact = 1;
Ourmat = A;

params.meas_lambda = lambda;
%note the 2*hatw this may need to be changed
%Also the l1 and l2 order may be wrong
[ soln ] = run_example( Ourmat, hatw, 'l1', 'l2', [], params );

%muahahahahaha
params.E = Ourmat; 
params.e = hatw; 
params.eqFlag =1; % do we need this?
tic
[aff_soln] = run_example_affine(D, zeros(Dr,1), 'huber', [], [], params);
time = toc;


%extracting states from soln
Dstateblock = [sparse(n,n) sparse(n,m) speye(n,n)];
Dstate = blktridiag(Dstateblock,sparse(n, 2*n+m), sparse(n, 2*n+m),N);
Dstate = [speye(n,n) sparse(n, N*(2*n +m)); sparse(N*n, n) Dstate];
statesoln = Dstate*soln;
%reshaping state solution to have desired form
statesoln = reshape(statesoln, n, N+1);

%extracting w's from soln (these are actually the u's in z in the writeup)
wblock = [speye(n,n), sparse(n,m), sparse(n,n)];
wextract = blktridiag(wblock, sparse(n, 2*n + m), sparse(n, 2*n+m),N);
wextract = [sparse(n*N, n), wextract];
wsoln = wextract*soln;
%reshaping
wsoln = reshape(wsoln,n, N);

%finally extracting v's from soln (these are the t's in z in the writeup)
vblock = [sparse(m,n), speye(m,m), sparse(m,n)];
vextract = blktridiag(vblock, sparse(m, 2*n+m), sparse(m, 2*n+m), N);
vextract = [sparse(m*N, n), vextract];
vsoln = vextract*soln;
%reshaping
vsoln = reshape(vsoln,m, N);

if ~exist('othersoln', 'var')
    othersoln = 'state';
end

if strcmp(othersoln, 'state')
    soln = statesoln;
elseif strcmp(othersoln, 'w')
    soln = wsoln;
elseif strcmp(othersoln, 'v')
    soln = vsoln;
end


end

