function[xopt] = SingularSystemSolver(A, hatw, c,At2,q)
%solves system of the form:
%(I  A^T) (   x  )  = c
%(A   0 ) (lambda)  = hatw

%useNormal = 1; 
%[m,n] = size(A);


% Mat = [speye(n), A'; A sparse(m,m)];
% rhs = [c; hatw];
% sol = Mat\rhs;
% xopt = sol(1:n);


%if(useNormal)
 %   Mat = -1*(A*A');
  %  b = hatw - A*c;
  %  lambda = Mat\b;
%    xopt = c - A*lambda;
%    xopt = c + A*(A*A')\(hatw - A*c)
%    xopt = (I-A*(A*A')^{-1}*A')*c + A'*(A*A')\hatw; 
t1 = c-q*(q'*c);
xoptMy = t1+At2;
%    xoptMy = c-q*(q'*c) + A'*((A*A')\hatw);
else
    Mat = [speye(n), A'; A sparse(m,m)];
    rhs = [c; hatw];
    sol = Mat\rhs;
    xopt = sol(1:n);
end