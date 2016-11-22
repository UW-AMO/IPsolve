%% Kalman example: exploiting matrix structure!

clear all; close all; clc

signal = 'exp';
switch(signal)
    
    case{'sine'}
        gamma =  1;         % transition covariance multiplier
        signalFunc = @(x) sin(-x);
        box = [-1,1]; % lower, upper 
        numP = 4;
        N     = 100;        % number of measurement time points
        dt    = numP*2*pi / N;  % time between measurement points
        sigma =  .4;       % standard deviation of measurement noise
        sigmaMod = 0.1;    % sigma we tell smoother
        outliers = 1;
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
n     = 2;          % number of measurements per time point
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
v_min   = min(x_true)*1.3 -1;    % minimum vertical value in plots
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
rk      = (sigma * sigma);
rinvk   = 1 / rk;

% create measurement model
rh = sqrt(rinvk)*[0, 1];
Hmat = kron(speye(N), rh);
meas = sqrt(rinvk)*z'; % scaling everyone


%% process model
qk      = gamma * [ dt , dt^2/2 ; dt^2/2 , dt^3/3 ];
multMat = (sqrt(qk)\eye(n));
gk = multMat*[ 1 , 0 ; dt , 1 ];

%lowd = -2*ones(2);
Gmat = blktridiag(multMat*speye(n),-gk, zeros(n),N);
w = zeros(n*N, 1);
w(1:n) = 10*x_true(:,1);
Gmat(1:n, 1:n) = 10*eye(n); % initial state estimate 
params.procLinear = 1;
params.K = Gmat;
params.k = w;


%%

% regular kalman smoother
tic
[ yOut ~] = run_example( Hmat, meas, 'l2', 'l2', [], params );
toc

xOut = reshape(yOut, n, N); % nominal estimate


% Robust Kalman smoother - the code will automatically plot it.
% Huber huber!!
tic
[ yOut ~] = run_example( Hmat, meas, measPLQ, procPLQ, [], params );
toc

xOutRobust = reshape(yOut, n, N);


params.constraints = 1;
A = kron(speye(N), conA);
a = kron(ones(N,1), cona);
params.A = A'; % historical artifact
params.a = a; 

tic
[ yOut ~] = run_example( Hmat, meas, 'l2', 'l2', [], params );
toc
xOutCon = reshape(yOut, n, N); % constrained estimate

tic
[ yOut ~] = run_example( Hmat, meas, measPLQ, procPLQ, [], params );
toc

xOutConRobust = reshape(yOut, n, N);



% if draw_plot
figure(1);
clf
hold on
plot(t', x_true', 'r-', 'Linewidth',2 );
plot(t', z(1,:)', 'ko', 'Linewidth', 2);
plot(t', xOut(2,:)', 'b--', 'Linewidth', 2);
plot(t', xOutRobust(2,:)', 'm-.', 'Linewidth',2 );

%     plot(t', - ones(N,1), 'b-');
%      plot(t', ones(N,1), 'b-');
axis([h_min, h_max, v_min, v_max]);
title('Unconstrained Estimates');
legend('Truth', 'Meas.', 'Least Squares', 'Robust', 'Location', 'SouthEast' );
hold off
%

figure(2);
clf
hold on
plot(t', x_true', 'r-', 'Linewidth',2 );
plot(t', z(1,:)', 'ko', 'Linewidth', 2);
plot(t', xOutCon(2,:)', 'b--', 'Linewidth', 2);
plot(t', xOutConRobust(2,:)', 'm-.', 'Linewidth',2 );

%     plot(t', - ones(N,1), 'b-');
%      plot(t', ones(N,1), 'b-');
axis([h_min, h_max, v_min, v_max]);
title('Constrained Estimates');
legend('Truth', 'Measurements', 'Estimate', 'robust estimate', 'Location', 'SouthEast' );
hold off
%

print('KalmanComp','-depsc')
%savefig(1, '/Users/saravkin11/Dropbox/JonathanSasha/Notes/figures/');

% end
%
%

%end