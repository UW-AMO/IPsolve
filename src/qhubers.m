function [f,g] = qhubers(x, thresh, tau)
% thresh is the threshold for the quadratic part of the penalty
% tau is the quantile

if(nargin < 3)
   tau = 1;  
end

g = zeros(size(x));

left = (x <= -2*tau*thresh);
center = ((x > -2*tau*thresh) & (x < 2*(1-tau)*thresh));
right = (x >= 2*(1-tau)*thresh);

fleft = sum(2*tau*abs(x(left)) - 2*thresh*tau^2);
g(left) = -2*tau;

fcenter = 0.5*sum(x(center).*x(center)/(thresh));
g(center) = x(center)/thresh;

fright = sum(2*(1-tau)*abs(x(right)) - 2*thresh*(1-tau)^2);
g(right) = 2*(1-tau);

f = fleft + fcenter + fright;
end