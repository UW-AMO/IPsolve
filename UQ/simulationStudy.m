
clear all
clf
clc
rand("seed", 5) %need to set both of these in octave
randn("seed", 10) %see above

m = 50;  % num of of rows (obs)
n = 10;   % num of columns (features)

k = 4; %number of zero entries at the solution

%check condition number
A = randn(m,n) +  10*binornd(1,0.5,m,n);

xTrue = 10*rand(n,1) + 2*ones(n,1); % uniform non-negative guy. 
xTrue(1:k) = 0;

bTrue = A*xTrue;
params.A = A;

% define constraints:
% -x <= 0
C = -speye(n); 
d = zeros(n,1); 
params.C = C; 
params.d = d; 
params.Ftol = 1e-10;
params.muTol = 1e-8; 
params.maxItr = 100;
numSims = 1;


XSOL = zeros(n,numSims);

for sim = 1:20
  % generate noisy obs
  b = A*xTrue +  20*randn(m,1);
  norm(bTrue-b)/norm(bTrue); %about 8percent relative difference.
  params.b = b;
  %initial guess
  xIn = 5*rand(n,1).*ones(n,1);
  
  % call solver
  [xSol, IMat] = sqIP(xIn, params); 
  %store solution
  XSOL(:,sim) = xSol;
endfor
format short
%XSOL
%xTrue
cov(XSOL') %have to transpose as MATLAB wants the rows to be observations.


%fprintf('dist to xTrue: %7.2e\n', norm(xSol - xTrue)/norm(xTrue));
%norm(xSol(1:k))
%[V,D] = eig(IMat);


%gv = linspace(-10,10,1000);
%[xx, yy]=meshgrid(gv, gv);
%F = D(1,1)*xx.*xx + D(2,2)*yy.*yy + 2*D(1,2)*xx.*yy;
%figure
%contour(xx,yy,F)
%title("Contours of Interior Point Precision Matrix")
%hold on
%plot(xTrue(1), xTrue(2))