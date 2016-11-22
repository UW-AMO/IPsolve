
clear all
clf
clc
% nonneg-LS runner
%rng(5) %set seed MATLAB
%rand("seed", 5) %need to set both of these in octave
%randn("seed", 10) %see above
m = 50;  % num of of rows (obs)
n = 2;   % num of columns (features)

k = 1; %number of zero entries at the solution

%check condition number
A = randn(m,n); % +  10*binornd(1,0.5,m,n);


xTrue = .1*rand(n,1) + ones(n,1); % uniform non-negative guy. 
xTrue(1:k) = 0;

sig = 1; 

b = A*xTrue +  sig*randn(m,1);

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
%%
xIn = 0*rand(n,1); 

params.maxItr = 2;
% call solver
[xSol, IMat, sSol, lamSol] = sqIP(xIn, params); 


% k x k singular values:
format long
% S = zeros(n,1);
% S(1:k)=svd(inv(IMat(1:k,1:k))); %get to bdry not changing
Sall = svd(inv(IMat)); 
%plot(Sall,'g') %svd of NEW in GREEN
ylabel('Singular Values of Estimated Covariances')
%hold on
%plot(svd(inv(A'*A)),'b') %svd of OLS in BLUE.
% plot(S,'r')
fprintf('dist to xTrue: %7.2e\n', norm(xSol - xTrue)/norm(xTrue));
norm(xSol(1:k))
%% show principal eigenvectors of the information

[V,D] = eig(IMat);
iIMat = inv(IMat);
ilsMat = inv(A'*A);

gv = linspace(-10,10,1000);
[xx, yy]=meshgrid(gv, gv);
F = iIMat(1,1)*xx.*xx +iIMat(2,2)*yy.*yy +2*iIMat(1,2)*xx.*yy;
G = ilsMat(1,1)*xx.*xx +ilsMat(2,2)*yy.*yy +2*ilsMat(1,2)*xx.*yy;
figure
contour(xx+xSol(1),yy+xSol(2),G,'--ro', 'LineWidth', 4)
%title('Contours of Interior Point Precision Matrix')
hold on
contour(xx+xSol(1),yy+xSol(2),F, '-.bo', 'LineWidth', 2)

plot(xTrue(1), xTrue(2), '^r', 'MarkerSize', 10);
plot(xSol(1), xSol(2), '*b', 'MarkerSize', 10);
y1 = get(gca, 'ylim');
x1 = get(gca, 'xlim');
plot([0 0], y1, 'LineWidth', 3);
plot(x1, [0 0], 'LineWidth', 3);
%axis equal;
axis([0, 10, 0, 10])