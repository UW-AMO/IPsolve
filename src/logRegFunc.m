function [ g H] = logRegFunc(u)
%QUADFUNC Summary of this function goes here
%   Detailed explanation goes here


if(u <= 0)
    error('u <= 0'); 
end

if(u >= 1)
    error('u>=1');
end


%f = sum(u.*log(u) + (1-u).*log(1-u)) + lam/2 *sum(u.*u);


%Regularization 
lam = 0; 

% Gradient
%g = log(u./(1-u)) + lam*u;
g = log(u) - log(1-u) + lam*u;

% Hessian
H = spdiags(1./(u.*(1-u)), 0, length(u), length(u)) + lam*speye(length(u));
    


end

