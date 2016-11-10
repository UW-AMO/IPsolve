% sine example using CP
%Here the state x = [position; velocity] and we model the acceleration as
%the error. As acceleration impacts position and velocity this leads to a
%singular covariance matrix in the state equations
%
%We model the acceleration with white mean zero noise. This should work
%well in this case
%

gamma = 1;
%signalFunc = @(x) sin(-x);
signalFunc = @(x) exp(sin(4*x))
box = [-1,1]; % lower, upper 
numP = 4;
N     = 5;        % number of measurement time points
dt    = numP*2*pi / (4*N);  % time between measurement points
sigma =  .1;       % standard deviation of measurement noise
sigmaMod = 0.1;    % sigma we tell smoot
outliers = 0;

if(outliers)
    out = N*.1;        % percent of outliers
    mag = 5;            % outlier variance
else
    out = 0;
    mag = 0;
end

m     = 1;          % number of measurements per time point
n     = 2;          % number of states per time point
t       =  (1 : N) * dt; % set of times

rand('seed', 1234);
%x1_true = - cos(t);
x_true = signalFunc(t);

%x_true  = [ x1_true ; x2_true ];
%
% measurement values and model
gaussErrors  = sigma * randn(1, N);
% construct the outliers
outliers = zeros(1,N);
inds = randperm(N, out); % indices of outliers
outliers(inds) = mag*randn(1,out);
z       = x_true + gaussErrors + outliers;

R = sigma*sigma;
gain = [.5*dt*dt; dt];
Q = gain*gain';
%Q = gamma^2*Q;
Q = 50^2/12*Q;

G = [1 dt; 0 1];
H = [1 0];

[D, A, hatw] = SingularKalmanSetup(G, H, Q, R, [0;0], N, z');
[Dr Dc] = size(D);


xopt = CPSolver(D, A, hatw, sparse(Dc, 1), sparse(Dr, 1));

plot(xopt)