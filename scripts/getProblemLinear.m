function [ params ] = getProblemLinear( pNum, params )
%GETPROBLEM sets parameter values depending on pNum
%

%test       = 1,2,3,4,5;   % Simple linear test problem
% 2, 3 and 5 are degenerate. 4 has non-zero Fs
%test       = 7          % Approximate the values of f(z)=e^{-z}(z) taken
%                          for z:=0.0 step 0.02 until 4.0 (i.e. on 201 points)
%                           by a polynomial of degree n- 1.
%test       = 8          % Approximate the values of f(z)=e^{z} taken
%                          for z:=0.0 step 0.01 until 2.0 (i.e. on 201 points)
%                           by a polynomial of degree n- 1.
%test       = 10;        % Simple quadratric test set with exact gradients and Hessians
%test       = 11;        % Simple quadratric test set with finite difference gradients and
%                         quasi-Newton (SR1) Hessian updates.
%test       = 12;        % Simple non-quadratric test set with exact gradients and Hessians
%test       = 13;        % Simple non-quadratric test set finite difference gradients and
%                         quasi-Newton (SR1) Hessian updates.
%test       = 14;        % Simple non-quadratric test set finite difference gradients and
%                         quasi-Newton (SR1) Hessian updates.

switch pNum
    case {1}
        params.x0 = 0.5*ones(3,1);
        params.AA = [1 4 2;
            -1 2 3];
        params.b = [1 2]';
        params.lambda = .1;

        
    case {2}
        params.x0 = 0.5*ones(9,1);
        params.AA = [8 32 8 0 0 0 0;
            1 23 23 1 0 0 0;
            0 8 32 8 0 0 0;
            0 1 23 23 1 0 0;
            0 0 8 32 8 0 0;
            0 0 1 23 23 1 0;
            0 0 0 8 32 8 0;
            0 0 0 1 23 23 1;
            0 0 0 0 8 32 8]';
        params.b = [2, 1, 0, 0, 0, 1, 2 ]';
        params.lambda = .1;

    case {3}
        params.x0 = 0.5*ones(7,1);
        AA = [8 32 8 0 0 0 0;
            1 23 23 1 0 0 0;
            0 8 32 8 0 0 0;
            0 1 23 23 1 0 0;
            0 0 8 32 8 0 0;
            0 0 1 23 23 1 0;
            0 0 0 8 32 8 0;
            0 0 0 1 23 23 1;
            0 0 0 0 8 32 8];
        BB = [1 -2 1 0 0 0 0;
            0 1 -2 1 0 0 0;
            0 0 1 -2 1 0 0;
            0 0 0 1 -2 1 0;
            0 0 0 0 1 -2 1];
        params.AA = [AA; BB];
        params.b = [2, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0]';
        params.lambda = .1;

    case{4}
        params.x0 = 0.5*ones(3,1);
        params.AA = [1 4 2;
            -1 2 3;
            0 6 5];
        params.b = [1 2 -1]';
        params.lambda = .1;

    case{5}
        params.x0 = 0.5*ones(3,1);
        params.AA = [1 4 2;
            -1 2 3;
            0 6 5];
        params.b = [1 2 3]';
        params.lambda = .1;

        
        
    case {6}
        params.x0 = 0.5*ones(5,1);
        params.AA = [5  3  4  12 4;
            9  7  3  19 13;
            6  6  0  12 12;
            9  9  7  25 11;
            3  0  1  4  2;
            8  1  8  17 1;
            1  9  8  18 2;
            3  1  1  5  3;
            0  9  3  12 6 ];
        params.b= [7, 4, 2, 7, 7, 7, 3, 5, 3]';
        params.lambda = .1;

        
    case {7}
        nnn=8;
        params.x0 = zeros(nnn,1);
        mmm= 201;
        step = .02;
        up = step*(mmm-1);
        z = 0:step:up;
        params.AA = diag(z)*ones(mmm,nnn);
        params.AA(:,1) = ones(mmm,1);
        b = exp(-z).*z;
        params.b = b';
        params.lambda = .1;

        
        
    case {8}
        nnn=7;
        params.x0 = zeros(nnn,1);
        mmm= 201;
        step = .01;
        up = step*(mmm-1);
        z = 0:step:up;
        params.AA = diag(z)*ones(mmm,nnn);
        params.AA(:,1) = ones(mmm,1);
        b = exp(z);
        params.b = b';
        params.lambda = .1;

        
    case{9}
        
        m = 120; n = 1025; k = 20; % m rows, n cols, k nonzeros.
      %  m = 50; n = 500; k = 10; % m rows, n cols, k nonzeros.
        %m = 30; n = 100; k = 4; % m rows, n cols, k nonzeros.
        %params.x_0 = rand(n,1);
%        rng('default');
        p = randperm(n); x0 = zeros(n,1);
%        rng('default');
        x0(p(1:k)) = sign(randn(k,1));
        params.x0 = x0;
%        rng('default');
        A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
        params.AA = A;
%        rng('default');
        params.b  = A*x0 + 0.005 * randn(m,1);
        params.lambda = .1;

        
   
    case{10}
        m = 240; n = 2050; k = 40; % m rows, n cols, k nonzeros.
        %m = 120; n = 1025; k = 20; % m rows, n cols, k nonzeros.
        %m = 30; n = 100; k = 4; % m rows, n cols, k nonzeros.
        %m = 3; n = 10; k = 4; % m rows, n cols, k nonzeros.
        %rng('default');
        %params.x_0 = rand(n,1);
        params.x0 = zeros(n,1);
        rng('default');
        p = randperm(n); x0 = zeros(n,1);
        rng('default');
        x0(p(1:k)) = sign(randn(k,1));
        rng('default');
        A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
        params.AA = A;
        rng('default');
        params.b  = A*x0 + 0.005 * randn(m,1);
        params.lambda = .1;
        params.x0 = x0;

    
    case{11}
        m = 480; n = 4100; k = 80; % m rows, n cols, k nonzeros.
        %rng('default');
        %params.x_0 = rand(n,1);
        params.x0 = zeros(n,1);
        rng('default');
        p = randperm(n); x0 = zeros(n,1);
        rng('default');
        x0(p(1:k)) = sign(randn(k,1));
        rng('default');
        A  = randn(m,n); [Q,R] = qr(A',0);  A = Q';
        params.AA = A;
        rng('default');
        params.b  = A*x0 + 0.005 * randn(m,1);
        params.x0 = x0;
        params.lambda = .1;
                

end

