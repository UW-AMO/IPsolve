function [ z,  H ] = nonlinThree(x)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


F(1) =  x(1)^3+x(1) + 4*x(2) + 2*x(3)^2 - 1;
F(2) = x(2)*x(3)^2 -x(1) + 2*x(2) + 3*x(3) - 2;
J(1,1) =  3.0*x(1)*x(1)+1.0;
J(1,2) =  4.0;
J(1,3) =  4.0*x(3);
J(2,1) =  -1.0;
J(2,2) =  2.0+x(3)^2;
J(2,3) = 3.0 + 2.0*x(2)*x(3);
z = F';
H = J;

end

