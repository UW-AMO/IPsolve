%% Kalman example: exploiting matrix structure!

clear all; close all; clc

signal = 'sine';
switch(signal)
    
    case{'sine'}
        gamma =  1;         % transition covariance multiplier
        signalFunc = @(x) sin(-x);
        box = [-1,1]; % lower, upper 
        numP = 4;
        N     = 100;        % number of measurement time points
        dt    = numP*2*pi / N;  % time between measurement points
        sigma =  .1;       % standard deviation of measurement noise
        sigmaMod = 0.1;    % sigma we tell smoother
        outliers = 0;
        if(outliers)
            out = N*.05;        % percent of outliers
            mag = 5;            % outlier variance
        else
            out = 0;
            mag = 0;
        end
        
        % box constraints for constrained estimates
        conA = [0 1; 0 -1];
        cona = [1; 1];
        
        measPLQ = 'huber';
        procPLQ = 'l2';
        
        
    case{'exp'}
        gamma = 0.1;
        mult = 4;
        signalFunc = @(x) exp(sin(mult*x));
        numP  = mult;         %
        N     = 100;        % number of measurement time points
        dt    = numP*2*pi / (N*mult);  % time between measurement points
        sigma =  .05;       % standard deviation of measurement noise
        sigmaMod = sigma;   % give nominal variance 
        outliers = 1;
        if(outliers)
            out = N*.1;        % percent of outliers
            mag = 5;            % outlier variance
        else
            out = 0;
            mag = 0;
        end
        
        
        measPLQ = 'huber';
        procPLQ = 'huber';
        
                % box constraints
        conA = [0 1; 0 -1];
        cona = [exp(1); -exp(-1)];

        
        
    otherwise
        error('unknown signal');
end

m     = 1;          % number of measurements per time point
n     = 2;          % number of states per time point
t       =  (1 : N) * dt; % set of times


max_itr = 30;      % maximum number of iterations
epsilon = 1e-5;    % convergence criteria
% ---------------------------------------------------------



%% Generate state data
% smooth signal:

rand('seed', 1234);
%x1_true = - cos(t);
x_true = signalFunc(t);

% plotting parameters
h_min   = 0;       % minimum horizontal value in plots
h_max   = dt*N;       % maximum horizontal value in plots
v_min   = min(x_true)*1.3 -.2;    % minimum vertical value in plots
v_max   = max(x_true)*1.3 +.2;    % maximum vertical value in plots

%x_true  = [ x1_true ; x2_true ];
%
% measurement values and model
gaussErrors  = sigma * randn(1, N);
% construct the outliers
outliers = zeros(1,N);
inds = randperm(N, out); % indices of outliers
outliers(inds) = mag*randn(1,out);
z       = x_true + gaussErrors + outliers;


%% measurement model
rk      = (sigmaMod * sigmaMod);
rinvk   = 1 / rk;

% create measurement model
rh = sqrt(rinvk)*[0, 1];

Hmat = kron(speye(N), rh); % 
meas = sqrt(rinvk)*z'; % scaling everyone


%% process model
qk      = gamma * [ dt , dt^2/2 ; dt^2/2 , dt^3/3 ];
multMat = (sqrt(qk)\eye(n));
gk = multMat*[ 1 , 0 ; dt , 1 ];

%Full state (different)
Gmat = blktridiag(multMat*speye(n),-gk, zeros(n),N);
w = zeros(n*N, 1);
w(1:n) = 10*x_true(:,1);
Gmat(1:n, 1:n) = 10*eye(n); % initial state estimate 

fullMat = [Gmat; Hmat]; 
fullVec = [w; meas]; 
[mMat,nMat] = size(fullMat);

tic
yOut = lsqr(mMat,nMat, fullMat, fullVec, 0, 1e-10, 1e-10,1000, 1000, 1); 
timeLSQR = toc;


xOut = reshape(yOut, n, N); % nominal estimate


%% Singular smoother 
multMatSing = sqrt(qk);  % Q^{1/2]
multMatSing(2,2) = 0;
multMatSing(1,2) = 0; 
multMatSing(2,1) = 0;
multMatSing = sparse(multMatSing);

Rmat = sqrt(sigmaMod);   % R^{1/2} 
HH = [0, 1];
Gk = [ 1 , 0 ; dt , 1 ];

Diagblock = [ sparse(m, n), sparse(m,m), sparse(m,n);  multMatSing, sparse(n,m), speye(n)];
UDblock = [sparse(m, n), Rmat, HH; sparse(n,n), sparse(n,m), -Gk  ]; 
[dim1, dim2] = size(UDblock); 
Ourmat = blktridiag(Diagblock,UDblock, sparse(dim1,dim2),N);

[rownum, colnum] = size(Ourmat); 
newCol = sparse(rownum, n); 
newCol(m+1:m+n, 1:n) = -gk;
Ourmat = [newCol, Ourmat];
[rownum, colnum] = size(Ourmat); 
newRow = sparse(n, colnum); 
newRow(1:n, 1:n) = speye(n); 
newRow = [newRow];
Ourmat = [newRow; Ourmat]; 
anotherrow = [sparse(m,n+(N-1)*(2*n+m)) sparse(m,n) Rmat HH];
Ourmat = [Ourmat; anotherrow]; % A in the writeup
% ourMat is the constraint 

% make D to pull out u and t components of our variable
Dblock = blkdiag(sparse(n,n), speye(n,n), speye(m,m));
blockdim = 2*n + m;
D = blktridiag(Dblock, sparse(blockdim, blockdim), sparse(blockdim, blockdim),N);
D = [D sparse(N*(2*n + m),n)];
D = vertcat(D, sparse(n,N*(2*n + m) + n));

%need a way to make initial guess for x0
%here just use w0 = [0;0]
zmat = reshape(z', m,N);
zz = [sparse(n,N); zmat];
hatw = zz(:);
w0 = sparse(n,1);
hatw = [w0; sparse(m,1);hatw];



[Dr Dc] = size(D);
totalmat = [D; Ourmat];
[Ourmatr Ourmatc] = size(Ourmat);
totalmat = [totalmat vertcat(Ourmat', sparse(Ourmatr, Ourmatr))];

tic
soln = totalmat \ vertcat(sparse(Ourmatc, 1), hatw);
timeBS = toc;

%% Relaxed version 
lam = 1000;
StackMat = [D; lam*Ourmat]; 
stackB = [zeros(Dr,1); lam*hatw];
[mMat, nMat] = size(StackMat);
tic
yRelax = lsqr(mMat,nMat, StackMat, stackB, 0, 1e-8, 1e-8,1e12, 1000, 1); 
timeRelax = toc;


solnRelax = yRelax(1:Dr);

Dstateblock = [sparse(n,n) sparse(n,m) speye(n,n)];
Dstate = blktridiag(Dstateblock,sparse(n, 2*n+m), sparse(n, 2*n+m),N);
Dstate = [speye(n,n) sparse(n, N*(2*n +m)); sparse(N*n, n) Dstate];
relaxsoln = Dstate*solnRelax;

xOutRelax = reshape(relaxsoln, n, N+1); % nominal estimate
xOutRelax = xOutRelax(:,2:length(xOutRelax));

fprintf('time Relax: %7.2e\n', timeRelax);


%% IPsolve version
params.K = D;
params.k = zeros(Dr,1);
params.inexact = 1;
%params.meas_lambda = lam; % measurement lambda

% tic
% [ solnIPsolve ] = run_example( StackMat, stackB, 'l2', []', [], params );
% timeIP = toc

params.meas_lambda = 1e6;

params.K = D; 
params.k = zeros(Dr,1);
tic
[ solnIPsolve ] = run_example( Ourmat, 2*hatw, 'l2', 'l2', [], params );
timeIP = toc



solnIP = solnIPsolve(1:Dr);

Dstateblock = [sparse(n,n) sparse(n,m) speye(n,n)];
Dstate = blktridiag(Dstateblock,sparse(n, 2*n+m), sparse(n, 2*n+m),N);
Dstate = [speye(n,n) sparse(n, N*(2*n +m)); sparse(N*n, n) Dstate];
IPsoln = Dstate*solnIP;

xOutIP = reshape(IPsoln, n, N+1); % nominal estimate
xOutIP = xOutIP(:,2:length(xOutIP));

fprintf('time Relax: %7.2e\n', timeIP);

%%

fprintf('size: %d, time LSQR: %7.2e, time ours: %7.2e, time relaxed: %7.2e, time IP: %7.2e \n', N*n, timeLSQR, timeBS, timeRelax, timeIP); 



soln = soln(1:Dr);

Dstateblock = [sparse(n,n) sparse(n,m) speye(n,n)];
Dstate = blktridiag(Dstateblock,sparse(n, 2*n+m), sparse(n, 2*n+m),N);
Dstate = [speye(n,n) sparse(n, N*(2*n +m)); sparse(N*n, n) Dstate];
statesoln = Dstate*soln;




statesoln = reshape(statesoln, n, N+1);
%Now remove the initial state
statesoln = statesoln(:,2:length(statesoln));

plot(t', x_true, 'k', t', statesoln(2,:), 'r--o',t',  xOut(2,:), 'b--*',... 
    t', xOutRelax(2,:), 'm--+', t', xOutIP(2,:), 'c--s')
legend('True Solution', 'Singular Est.', 'Nonsingular Est', 'relaxed', 'IP')





% % if draw_plot
% figure(1);
% clf
% hold on
% plot(t', x_true', 'r-', 'Linewidth',2 );
% plot(t', z(1,:)', 'ko', 'Linewidth', 2);
% plot(t', xOut(2,:)', 'b--', 'Linewidth', 2);
% %plot(t', xOutRobust(2,:)', 'm-.', 'Linewidth',2 );
% 
% %     plot(t', - ones(N,1), 'b-');
% %      plot(t', ones(N,1), 'b-');
% axis([h_min, h_max, v_min, v_max]);
% title('Unconstrained Estimates');
% legend('Truth', 'Meas.', 'Least Squares', 'Location', 'SouthEast' );
% hold off
% %




