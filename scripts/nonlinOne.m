function [ z,  H ] = nonlinOne(x)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

F(1) =  x(4)- 0.5*x(1)^2 -x(2)^2 - 1.5*x(3)^2 -x(1) -x(2) -x(3);
F(2) = x(1)*x(2) + x(1)*x(3) + x(2)*x(3) - 1;
F(3) = x(1)^2 + x(2)^2 + x(3)^2 -2 +0.6388;
F(4) = 3.0*x(1) + 2.0*x(2) + x(3)-13.0/3.0;
F(5) = x(1)*x(2) + 0.5*x(3)*x(3) - 1 + 0.4444444;
%Quadratics/Simple non-quadratics with exact gradient and Hessian information
J(1,1) =-x(1) -1.0;
J(1,2) = -2.0*x(2) -1.0;
J(1,3) = -3.0*x(3) -1.0;
J(1,4) = 1.0;
J(2,1) = x(2) + x(3);
J(2,2) = x(1) + x(3);
J(2,3) = x(1) + x(2);
J(2,4) = 0.0;
J(3,1) = 2.0*x(1);
J(3,2) = 2.0*x(2);
J(3,3) = 2.0*x(3);
J(3,4) = 0.0;
J(4,1) = 3.0;
J(4,2) = 2.0;
J(4,3) = 1.0;
J(4,4) = 0.0;
J(5,1) = x(2);
J(5,2) = x(1);
J(5,3) = x(3);
J(5,4) = 0.0;

z = F';
H = J;

end

