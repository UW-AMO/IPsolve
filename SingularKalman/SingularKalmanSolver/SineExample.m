%Sine example

gamma = 1;
signalFunc = @(x) sin(-x);
box = [-1,1]; % lower, upper 
numP = 4;
N     = 100;        % number of measurement time points
dt    = numP*2*pi / N;  % time between measurement points
sigma =  .1;       % standard deviation of measurement noise
sigmaMod = 0.1;    % sigma we tell smoot
outliers = 0;

if(outliers)
    out = N*.05;        % percent of outliers
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



R = (sigmaMod * sigmaMod);
Q      = gamma * [ dt , 0 ; 0 , 0 ];

H = [0, 1];
G = [ 1 , 0 ; dt , 1 ];

[D, A, hatw] = SingularKalmanSetup(G, H, Q, R, [0;0], N, z');
[Dr Dc] = size(D);
Lagsoln = LagrangianSolver(A, D, hatw, n,m, N);
%Now remove the initial state
Lagsoln = Lagsoln(:,2:length(Lagsoln));

%muahahahahaha
params.E = A; 
params.e = hatw; 
params.eqFlag =1; % do we need this?
tic
[aff_soln] = run_example_affine(D, zeros(Dr,1), 'huber', [], [], params);
time = toc;

[xsoln, vsoln, wsoln] = extractor(m, n, N, aff_soln);

plot(t', x_true, 'k', t', xsoln(2,2:end)', 'r--o', t', z, 'b*')
legend('truth', 'singular estimate');

%[Lagsoln, Relaxsoln, Robustsoln] = SingularKalmanRunner(G, H, Q, R, [0;0], N, z', 100);


