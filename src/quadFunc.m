function [ g, H] = quadFunc( u, M )
%QUADFUNC Summary of this function goes here
%   Detailed explanation goes here
g = M*u;
if(nargout > 1)
   
   H = M; 
end

end

