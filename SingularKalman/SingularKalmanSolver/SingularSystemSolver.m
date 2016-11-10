function[xopt] = SingularSystemSolver(A, hatw, c)
%solves system of the form:
%(I  A^T) (   x  )  = c
%(A   0 ) (lambda)  = hatw


Mat = -1*A*A';
b = hatw - A*c;
lambda = Mat\b;
xopt = c - A'*lambda;
