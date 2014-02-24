% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [ M C c b B fun] = loadPenalty( H, z, penalty, params )
%loadPENALTY Loads one of several predefined penalties, 
% for IP solver. 

m = params.size;

switch(penalty)

    case 'studentPL'
        scale    = params.scale;
        kappa = params.kappa;
        C = [speye(m); -speye(m)];
        c = kappa*ones(2*m, 1);
        M    = @(x)studentFunc(x, scale);
        b    = zeros(m, 1);
        B    = speye(m);

    
    case 'student'
        scale    = params.scale;
        M    = @(x)studentFunc(x, scale);
        C    = zeros(1,m); % easy to satisfy 0'*u <= 1
        c    = 1;
        b    = zeros(m, 1);
        B    = speye(m);

    
    case 'hybrid'
        scale    = params.scale;
        M    = @(x)hybridFunc(x, scale);
        C    = zeros(1,m); % easy to satisfy 0'*u <= 1
        c    = 1;
        b    = zeros(m, 1);
        B    = speye(m);
        
        % function handle to evaluate objective
        fun = @(x) sum(sqrt(1 + x.^2/scale) - 1);
        
        % function handle for gradients, in case this is needed elsewhere
        params.gfun = @(x) x./sqrt(1 + x.^2/scale);
    
    case 'vapnik'
        lam = params.lambda;
        eps = params.eps;
        M = 0*speye(2*m);
        %Cp = [speye(m) 0*speye(m); 0*speye(m) -speye(m)];
        %C = [Cp; Cp];
        %c = [ones(m, 1); zeros(m,1); ones(m,1); zeros(m,1)];
        C = [speye(2*m); -speye(2*m)];
        c = [lam*ones(2*m, 1); zeros(2*m,1)]; % vapnik!
       
        
        b = -eps*ones(2*m,1);
        B = [speye(m); -speye(m)];
        
        % function handle to evaluate objective
        fun = @(x) sum(lam*pos(x-eps) + lam*pos(-x-eps));
        
        
    
    case 'huber'
        kappa = params.kappa;
        mMult = params.mMult;
        M = mMult*kappa*speye(m);
        C = [speye(m); -speye(m)];
        c = ones(2*m, 1);
        b = zeros(m,1);
        B = speye(m);     
        
        
        % function handle to evaluate objective
        fun = @(x) sum((abs(x) > mMult*kappa).*(abs(x) - 0.5*mMult*kappa).*sign(x) + (abs(x) < 0.5*mMult*kappa).*x.^2/(mMult*kappa)); 
        
        % function handle for gradients, in case this is needed elsewhere
        params.gfun = @(x) (abs(x) > mMult*kappa).*sign(x) + 2*(abs(x) <= 0.5*mMult*kappa).*x/(mMult*kappa);
        
    case 'l1'
        lam = params.lambda;
        M = 0*speye(m);
        C = [speye(m); -speye(m)];
        c = lam*ones(2*m, 1);
        b = zeros(m,1);
        B = speye(m); 
        
        % function handle to evaluate objective
        fun = @(x) lam*norm(x,1);
        
        
    case 'qreg' % lambda-scaled penalty.
        lam = params.lambda;
        tau = params.tau;
        M = 0*speye(m);
        C = [speye(m); -speye(m)];
        c = lam*[(1-tau)*ones(m,1); tau*ones(m,1)];
%        c = lam*ones(2*m, 1);
        b = zeros(m,1);
        B = speye(m);   
        
        % function handle to evaluate objective
        fun = @(x) lam*tau*sum(pos(-x)) + lam*(1-tau)*sum(pos(x));

    case 'qhuber' % quantile huber penalty.
        tau = params.tau;
        kappa = params.kappa;
        M = kappa*speye(m);
        
        C = 0.5*[speye(m); -speye(m)];
        c = [(1-tau)*ones(m,1); tau*ones(m,1)];
        b = zeros(m,1);
        B = speye(m);           
        
%        fun = @(x) sum((x < -tau*kappa).*(-2*tau*x - kappa*tau^2) + x.^2/kappa + (x > (1-tau)*kappa).*(2*(1-tau)*x - kappa*(1-tau)^2));
        fun = @(x)qhubers(x, kappa, tau);
%sum(2*tau*abs(x(left)) - 2*thresh*tau^2)
        
    case 'l2func'
        mMult    = params.mMult;
        M    = @(x)quadFunc(x, mMult*speye(m));
        C    = zeros(1,m); % easy to satisfy 0'*u <= 1
        c    = 1;
        b    = zeros(m, 1);
        B    = speye(m);
               
    case 'l2'
        mMult    = params.mMult;
        M    = mMult*speye(m);
        C    = zeros(1,m); % easy to satisfy 0'*u <= 1
        c    = 1;
        b    = zeros(m, 1);
        B    = speye(m);

        % function handle to evaluate objective
        fun = @(x) norm(x)^2/(2*mMult);
        
    case 'hinge'
        M = 0*speye(m);
        lam = params.lambda;
        C = [speye(m); -speye(m)];
        c = lam*[ones(m, 1); zeros(m,1)]; % hinge!
        b = zeros(m,1);
        B = speye(m);
        
        % function handle to evaluate objective
        fun = @(x) lam*sum(pos(x));
        
              
    case 'l2m' % penalize everybody except the last element
        M = speye(m);
        C    = [zeros(2,m-1) ones(2,1)]; 
        c    = zeros(2,1);
        b    = zeros(m, 1);
        B    = speye(m);

        % function handle to evaluate objective
        fun = @(x) 0.5*norm(x(1:end-1))^2;
        
    case 'l1m'
        lam = params.lambda;
        M = 0*speye(m);
        C = [speye(m); -speye(m); zeros(2,m-1) ones(2,1)];
        c = [lam*ones(2*m, 1); zeros(2,1)];
        b = zeros(m,1);
        B = speye(m); 

        % function handle to evaluate objective
        fun = @(x) lam*norm(x(1:end-1), 1); 
        
    otherwise
        error('unknown PLQ');
end

  % composing with linear model
        %b = b-B*z; 
         b = -b+B*z;
        %B = B*H; 
        B = -B*H;
                
end

