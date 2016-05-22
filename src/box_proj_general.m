function [p, g]= box_proj_general(y, high, low)
% input: y:     vector of inputs
%        high:  upper limit of box
%        low:   lower limit of box

n = length(y); 
p = max(min(y, high), low); 

if(nargout > 1)
%     if ((any(y==high)) || (any(y==low)))
%         disp('y==high');
%     end
%     preg = (p==y);
%     preg(y==high) = 1/2;
%     preg(y==low) = 1/2;
%     g = sparse(1:n,1:n, preg);
    g = sparse(1:n, 1:n, p==y);
    % if you didn't change, jacobian = 1
    % if you were clipped, jacobian = 0. 
    
    %spdiags(preg, 0, n, n);
%     g = spdiags((y < high) .* (y > low), 0, n, n);
end