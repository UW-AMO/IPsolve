%Checking the projections of our solver

gamma = 1;
signalFunc = @(x) sin(x);
%signalFunc = @(x) exp(sin(4*x))
box = [-1,1]; % lower, upper 
numP = 4;
N     = 50;        % number of measurement time points
dt    = numP*2*pi / (N);  % time between measurement points
sigma =  .1;       % standard deviation of measurement noise
sigmaMod = 0.1;    % sigma we tell smoot
outliers = 1;

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


G = [1 dt; 0 1];
H = [1 0];

[D, A, hatw] = SingularKalmanSetup(G, H, Q, R, [0;0], N, z');
[Dr Dc] = size(D);

params.E = A; 
params.e = hatw; 
params.eqFlag =1; % do we need this?
tic
[aff_soln] = run_example_affine(D, zeros(Dr,1), 'huber', [], [], params);
time = toc;

[xsoln, vsoln, wsoln] = extractor(m, n, N, aff_soln);

h_min   = 0;       % minimum horizontal value in plots
h_max   = dt*N;       % maximum horizontal value in plots
v_min   = min(x_true)*1.3 -1;    % minimum vertical value in plots
v_max   = max(x_true)*1.3 +.2;    % maximum vertical value in plots
Proj = 1/(.25*dt^4+dt^2)*(gain*gain');
for k =1:N-1
    ourcheck(k) = norm((eye(2,2) - Proj)*(xsoln(:,k+1) - G*xsoln(:,k)));
end
plot(1:N-1, ourcheck)


% plot(t', x_true, 'k', t', xsoln(1,2:end)', 'r--', t', z, 'ko', 'Linewidth', 2)
% legend('truth', 'singular estimate','observed data');
% axis([h_min, h_max, v_min, v_max]);