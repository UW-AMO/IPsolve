function [p, g]= l1_prox_general(y, gamma, lambda)
% input: 
%  y:      vector of inputs
%  gamma:  multiplier of 1-norm


n = length(y); 
p = sign(y).*max(abs(y)-gamma*lambda,0);

if(nargout > 1)
    ap = abs(sign(p));
    g = sparse(1:n, 1:n, ap);
%     g = spdiags((y < high) .* (y > low), 0, n, n);
end