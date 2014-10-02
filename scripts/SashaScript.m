% requires cvx
test = 3;

switch test
    case{2}
        
        b= [2, 1, 0, 0, 0, 1, 2]';
        
        A= [8 32 8 0 0 0 0;
            1 23 23 1 0 0 0;
            0 8 32 8 0 0 0;
            0 1 23 23 1 0 0;
            0 0 8 32 8 0 0;
            0 0 1 23 23 1 0;
            0 0 0 8 32 8 0;
            0 0 0 1 23 23 1;
            0 0 0 0 8 32 8]';
    case{3}
        
        b= [2, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0]';
        
        A= [8 32 8 0 0 0 0;
            1 23 23 1 0 0 0;
            0 8 32 8 0 0 0;
            0 1 23 23 1 0 0;
            0 0 8 32 8 0 0;
            0 0 1 23 23 1 0;
            0 0 0 8 32 8 0;
            0 0 0 1 23 23 1;
            0 0 0 0 8 32 8;
            1 -2 1 0 0 0 0;
            0 1 -2 1 0 0 0;
            0 0 1 -2 1 0 0;
            0 0 0 1 -2 1 0;
            0 0 0 0 1 -2 1];
    case {6}
        b= [7, 4, 2, 7, 7, 7, 3, 5, 3]';
        A = [5  3  4  12 4;
            9  7  3  19 13;
            6  6  0  12 12;
            9  9  7  25 11;
            3  0  1  4  2;
            8  1  8  17 1;
            1  9  8  18 2;
            3  1  1  5  3;
            0  9  3  12 6 ];
    case {7}
            nnn=8;
            mmm= 201;
            b = zeros(mmm,1);
            z=0.0;
            A = ones(mmm,nnn);
            for i = 1:mmm
                for j = 2:nnn
                     A(i,j) = A(i,j)*z;
                end
                b(i) = exp(-z)*z;
                z = z +0.02;
            end          
    case {8}
            nnn=7;
            mmm= 201;
            b = ones(mmm,1);
            z=0.0;
            A = ones(mmm,nnn);
            for i = 1:mmm
                for j = 2:nnn
                     A(i,j) = A(i,j)*z;
                end
                b(i) = exp(z);
                z = z +0.01;
            end          
end

m = size(A, 1);
n = size(A, 2);

mu = 1;

%%


% cvx_begin
% variables x(n)
% minimize( mu*norm(A*x -b, 1) + norm(x, 1) )
% cvx_end


cvx_begin 
  variables x(n) y(m) pPlus(n) pMinus(n)
  dual variable lambda
  dual variable gamma
  dual variable eta1
  dual variable eta2
  minimize( mu*norm(y, 1) + ones(1,n)*pPlus + ones(1,n)*pMinus ) 
  subject to 
    lambda: y == A*x - b; 
    gamma: x == pPlus - pMinus;
    eta1: 0 <= pPlus;
    eta2: 0 <= pMinus;
cvx_end 

% cvx_begin 
%   variables x(n) y(m) p(n) 
%   dual variable lambda
%   dual variable eta1
%   dual variable eta2
%   minimize( mu*norm(y, 1) + ones(1,n)*p) 
%   subject to 
%     lambda: y == b-A*x; 
%      eta1: -p <= x;
%      eta2: x <= p;
% cvx_end 
% 
% xMult = eta1-eta2;


% cvx_begin
% variables x(n)
% dual variable y
% minimize(norm(x, 1) )
% subject to
% A*x == b
% cvx_end
% 
% 
% cvx_begin
% variables x(n)
% dual variable y
% minimize(norm(x, 1) )
% subject to
% y: A*x == b
% cvx_end
