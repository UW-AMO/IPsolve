function [ params ] = getProblemNonlinear( pNum, params )
%GETPROBLEMNONLINEAR sets parameter values depending on pNum
%


switch pNum

                       
    case{1}
        % status: works great. 
        
        n = 4;
        k = 2;
        p = randperm(n); x0 = zeros(n,1);
        rng('default');
        x0(p(1:k)) = sign(randn(k,1));
        params.x0 = x0;

        
        [z] = nonlinOne(x0);
        params.AA = @nonlinOne;
        params.b = z;
        params.n = n;
        params.m = 5;
        params.lambda = .1;

        
        
    case{2}
        % status: doesn't work at all. 
        
        m = 2; 
        n = 6;  % m functions, n variables.
        params.x_0 = zeros(n,1);
        %rng('default');
        p = randperm(n); x0 = zeros(n,1);
        %rng('default');
        k = 2;
        x0(p(1:k)) = randn(k,1);
        
        params.x0 = x0;
        
        
        [z] = nonlinTwo(x0);
        params.AA = @nonlinTwo;
        params.b = z;
        params.n = n;
        params.m = m;
        params.lambda = 1*1e-1;

    case{3}
        % status: sometimes works. 
        
        m = 2; n = 3;  % m functions, n variables.
        params.x_0 = zeros(n,1); 
        %rng('default'); 
        k = 2;
        p = randperm(n); x0 = zeros(n,1);
        %rng('default');
        x0(p(1:k)) = randn(k,1);
        
        params.x0 = x0;
        
        
        [z] = nonlinThree(x0);
        params.AA = @nonlinThree;
        params.b = z;
        params.n = n;
        params.m = m;
        params.lambda = 1;
        
    case{4}
        % status: doesn't work at all
        rng('default');
        m = 4; n = 5;  % m functions, n variables.
        x0 = [2;sqrt(2);-1;2*-sqrt(2);0.5];
        
        
        inds = randperm(n,2);
        x0(inds) = 0;
         
        
        
        params.x0 = x0; %zeros(n,1);
        
        [z] = nonlinFour(x0);% + 0.01*randn(m,1);
        params.AA = @nonlinFour;
        params.b = z;
        params.n = n;
        params.m = m;
        params.lambda = 1e0;
        
    case{5}
        % status: sometimes works, especially if randomly initialized 
        
        m = 4; n = 4;  % m functions, n variables.
        %x0 = [0.8;0.8;0.8;0.8]; 
        x0 = rand(n,1);
        
         
        inds = randperm(n,2);
        x0(inds) = 0;
        
        params.x0 = x0;
        
        [z] = nonlinFive(x0);% + 0.01*randn(m,1);
        params.AA = @nonlinFive;
        params.b = z;
        params.n = n;
        params.m = m;
        params.lambda = 1e-2;
        
    case{6}
        % status: sometimes recovers a spike, when initalized with zeros
        
        m = 3; n = 5;  % m functions, n variables.
        %x0 = [10;7;2;-3;0.8]; 
        x0 = randn(n,1);
       
         
        inds = randperm(n,4);
        x0(inds) = 0;
        
        params.x0 = x0;
        
        [z] = nonlinSix(x0);% + 0.01*randn(m,1);
        params.AA = @nonlinSix;
        params.b = z;
        params.n = n;
        params.m = m;
        params.lambda = 1e-2;
        
        
    case{7}
        % status: sometimes works, recovers
        m = 4; n = 5;  % m functions, n variables. 
        x0 = [2;sqrt(2);-1;2*-sqrt(2);0.5];
        
        inds = randperm(n,3);
        x0(inds) = 0;
        
        params.x0 = x0;
        
        [z] = nonlinSeven(x0);% + 0.01*randn(m,1);
        params.AA = @nonlinSeven;
        params.b = z;
        params.n = n;
        params.m = m;
        params.lambda = 1e-2;
        
    case{20}
        HelmRunner; % sets up all parameters
%        params.b = [real(params.F_0); imag(params.F_0)];
        params.b = real(params.F_0);
        params.x0 = x_0;
        params.AA = @(x)HelmFun(x, params); 
        params.n = size(x_0, 1);
        params.m = size(params.b, 1);
   end


end

