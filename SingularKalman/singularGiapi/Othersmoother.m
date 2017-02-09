%Testing other singular smoother

%%%%%%SINE EXAMPLE%%%%%
addpath(genpath(pwd))
clear all
gamma = 1;
signalFunc = @(x) sin(x);
box = [-1,1]; % lower, upper 
numP = 4;
% %<<<<<<< HEAD
% N     = 40;        % number of measurement time points
% =======
N     = 100;        % number of measurement time points
%>>>>>>> 1c9597f2799ed4051332a8343024a47a55da6987
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
%z = z +100*rand(size(z));

y = num2cell(z);
Rvect = sigma^2*ones(1,N);
R = num2cell(Rvect);
hvect = kron([1 0], ones(N,1));
h = num2cell(hvect,2);
ginst = [1 dt; 0 1];
mu = ginst*[0;1];
gain = [.5*dt*dt; dt];
Po = gain*gain';
for j=1:N
    g{j} = ginst;
    Q{j} = gain*gain';
end



%Xs e Ps: smoothed estimates and convariances

%%%%Kalman filter
%mu=x(1|0)
%Po=P(1|0)
%[mu,Po,y{1},R{1}]......(g{1},Q{1}).....[x2,y{2},R{2}].......(g{2},Q{2})...





warning off

n=length(y);%number of time instants where data are collected
nx=size(Po,1);%state space dimension


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% FORWARD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% TIME UPDATE
Xp(:,1)=mu;%stima stato predetta
Pp{1}=Po;%covarianza stato predetta
for i=1:n
    %%%%%%%%%%%%%%% MEASUREMENT UPDATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Ve{i}=h{i}*Pp{i}*h{i}'+R{i};%innovation variance
    e{i}=y{i}-h{i}*Xp(:,i);%innovation
    %Xf(:,i)=Xp(:,i)+Pp{i}*h{i}'*pinv(Ve{i})*e{i};
    Xf(:,i)=Xp(:,i)+Pp{i}*h{i}'*(Ve{i}\e{i});
    %Pf{i}=Pp{i}-Pp{i}*h{i}'*pinv(Ve{i})*h{i}*Pp{i};
    Pf{i}=Pp{i}-Pp{i}*h{i}'*(Ve{i}\h{i})*Pp{i};

    %%%%%%%%%%%%%%%%%%% TIME UPDATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if i<n
        Xp(:,i+1)=g{i}*Xf(:,i);
        Pp{i+1}=g{i}*Pf{i}*g{i}'+Q{i};
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% BACKWARD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% TIME UPDATE
Xs(:,n)=Xf(:,n);
Ps{n}=Pf{n};
for j=1:(n-1)
    i=n-j;
    %A{i}=Pf{i}*g{i}'*pinv(Pp{i+1});
    A{i}=((Pp{i+1}\g{i})*Pf{i})'; % assumes Pp are symmetric
    Xs(:,i)=Xf(:,i)+A{i}*(Xs(:,i+1)-Xp(:,i+1));
    Ps{i}=Pf{i}+A{i}*(Ps{i+1}-Pp{i+1})*A{i}';
end

%solutionps = PseudoSmoother(ginst, [1 0], z', Q{1}, R{1}, 2, 1, N, [0;1], 'l2');
%stateps = reshape(solutionps, 2, N);
%[M, P] = qr(Q{1});
M = Q{1};
% Proj = 1/(.25*dt^4+dt^2)*gain*gain';
% for k =1:n-1
%     check_f(k) = norm((eye(2) - Proj)*(Xf(:,k+1) - ginst*Xf(:,k)));
%     check_s(k) = norm((eye(2) - Proj)*(Xs(:,k+1) - ginst*Xs(:,k)));
%     %checkps(k) = norm((eye(2,2) - Proj)*(stateps(k+1) - ginst*stateps(k)));
% end
% % plot(1:n-1, check, 'r', 1:n-1, checkps, 'k')
%  plot(1:n-1, check_f, 'r')
%  hold on
%  plot(1:n-1, check_s, 'b')
% legend('Giapi projected filter error', 'Giapi projected smoother error')



[D, A, hatw] = SingularKalmanSetup(ginst, h{1}, Q{1}, R{1}, [0;1], N, z');
[Dr Dc] = size(D);

params.E = A; 
params.e = hatw; 
params.eqFlag =1; % do we need this?
tic
[aff_soln] = run_example_affine(D, zeros(Dr,1), 'l2', [], [], params);
time = toc;

[xsoln, vsoln, wsoln] = extractor(m, 2, N, aff_soln);
difference = norm(Xs(:,:) - xsoln(:, 2:end));
plot(t', Xs(1,:) - xsoln(1, 2:end))
% plot(t', Xs(1,:), 'b', t', x_true, 'r', t', stateps(1,:), 'k', 'Linewidth', 3)
% legend('Gianpi Solution', 'True Solution', 'IPSolve Solution') 
% hold on
% plot(t', outliers)