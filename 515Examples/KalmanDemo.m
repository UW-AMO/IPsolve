%% Kalman example: exploiting matrix structure! 

%clear all
N     = 100;        % number of measurement time points
dt    = 8*pi / N;  % time between measurement points
gamma =  1;        % transition covariance multiplier
sigma =  .05;       % standard deviation of measurement noise
max_itr = 30;      % maximum number of iterations
epsilon = 1e-5;    % convergence criteria
h_min   = 0;       % minimum horizontal value in plots
h_max   = dt*N;       % maximum horizontal value in plots
v_min   = -2.0;    % minimum vertical value in plots
v_max   = +2.0;    % maximum vertical value in plots
% ---------------------------------------------------------
ok = true;

%if nargin < 1
%    draw_plot = false;
%end


%  Define the problem
rand('seed', 1234);
%
% number of constraints per time point
ell   = 4;
%
% number of measurements per time point
m     = 1;
%
% number of state vector components per time point
n     = 2;
%
% simulate the true trajectory and measurement noise
t       =  (1 : N) * dt;
x1_true = - cos(t);
x2_true = - sin(t);
x_true  = [ x1_true ; x2_true ];
%
% measurement values and model
v_true  = sigma * randn(1, N);

out = 30;  % number of outliers 
mag = 3;   % outlier variance
% construct the outliers
outliers = zeros(1,N);
inds = randperm(N, out); % indices of outliers
outliers(inds) = mag*randn(1,out); 


z       = x2_true + v_true + outliers;
rk      = (sigma * sigma);
rinvk   = 1 / rk;

% create measurement model
rh = sqrt(rinvk)*[0, 1];

Hmat = kron(speye(N), rh); 
meas = sqrt(rinvk)*z'; % don't forget to scale everyone. 

% transition model
qk      = gamma * [ dt , dt^2/2 ; dt^2/2 , dt^3/3 ];
multMat = (sqrt(qk)\eye(n));
gk = multMat*[ 1 , 0 ; dt , 1 ];

%lowd = -2*ones(2);
Gmat = blktridiag(multMat*speye(n),-gk, zeros(n),N);
w = zeros(n*N, 1);
w(1:n) = 10*x_true(:,1);
Gmat(1:n, 1:n) = 10*eye(n);


%
% initial state estimate

% define linear process model
params.procLinear = 1;
params.K = Gmat;
params.k = w;


%%

% regular kalman smoother
tic
[ yOut ~] = run_example( Hmat, meas, 'l2', 'l2', [], params );
toc

xOut = reshape(yOut, n, N);



% Robust kalman smoother
params.kappa = 0.1;
tic
[ yOut ~] = run_example( Hmat, meas, 'huber', 'l2', [], params );
toc

xOutSparse = reshape(yOut, n, N);




% if draw_plot
    figure(1);
    clf
    hold on
    plot(t', x_true(2,:)', 'r-', 'Linewidth',2 );
    plot(t', z(1,:)', 'ko', 'Linewidth', 2);
    plot(t', xOut(2,:)', 'b--' );
    plot(t', xOutSparse(2,:)', 'm-.', 'Linewidth',2 );

    %     plot(t', - ones(N,1), 'b-');
    %      plot(t', ones(N,1), 'b-');
    axis([h_min, h_max, v_min, v_max]);
    title('Affine smoother');
    legend('Truth', 'Measurements', 'Estimate', 'robust estimate', 'Location', 'SouthEast' );
    hold off
    %
    % constrained estimate
    x_con = xOut;
    
    
    
    
% end
%    savefig(1, [pwd '/Experiments/affine']);
%

%end