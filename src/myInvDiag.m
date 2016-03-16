function A = myInvDiag(M) 
%input: diagonal matrix 
% output: its inverse
dM = diag(M); 
[s1, s2] = size(M);
A = spdiags(1./dM, 0, s1, s2);


end