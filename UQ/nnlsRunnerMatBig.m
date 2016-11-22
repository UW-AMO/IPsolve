
clear all
clf
clc
% nonneg-LS runner
%rng(5) %set seed MATLAB
%rand("seed", 5) %need to set both of these in octave
%randn("seed", 10) %see above
m = 10000;  % num of of rows (obs)
n = 50;   % num of columns (features)

k = 10; %number of zero entries at the solution

%check condition number
A = randn(m,n) +  10*binornd(1,0.5,m,n);


xTrue = 10*rand(n,1) + 2*ones(n,1); % uniform non-negative guy. 
xTrue(1:k) = 0;

b = A*xTrue +  2*randn(m,1);

% define constraints:
% -x <= 0
C = -speye(n); 
d = zeros(n,1); 

params.A = A;
params.b = b;
params.C = C; 
params.d = d; 
params.Ftol = 1e-10;
params.muTol = 1e-8; 

% Initialize
xIn = rand(n,1).*ones(n,1); 

% call solver
[xSol, IMat] = sqIP(xIn, params); 


% k x k singular values:
format long
 S = zeros(n,1);
 S(1:k)=svd(inv(IMat(1:k,1:k))); %get to bdry not changing
Sall = svd(inv(IMat)); 
plot(Sall,'g') %svd of NEW in GREEN
ylabel('Singular Values of Estimated Covariances')
hold on
plot(svd(inv(A'*A)),'b') %svd of OLS in BLUE.
 plot(S,'r')
fprintf('dist to xTrue: %7.2e\n', norm(xSol - xTrue)/norm(xTrue));
norm(xSol(1:k))
legend('New Covariance', 'OLS Variance', 'Jim stuff');
%% show principal eigenvectors of the information

[V,D] = eig(IMat);
iIMat = inv(IMat);

gv = linspace(-10,10,1000);
[xx, yy]=meshgrid(gv, gv);
%F = D(1,1)*xx.*xx + D(2,2)*yy.*yy + 2*D(1,2)*xx.*yy;
F = iIMat(1,1)*xx.*xx +iIMat(2,2)*yy.*yy +2*iIMat(1,2)*xx.*yy;
figure
contour(xx+xTrue(1),yy+xTrue(2),F)
title('Contours of Interior Point Precision Matrix')
hold on
plot(xTrue(1), xTrue(2), '*');