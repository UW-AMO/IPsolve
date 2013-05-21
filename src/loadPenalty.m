% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [ M C c b B ] = loadPenalty( H, z, penalty, params )
%loadPENALTY Loads one of several predefined penalties, 
% for IP solver. 

m = params.size;

switch(penalty)

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
    
    case 'huber'
        kappa = params.kappa;
        mMult = params.mMult;
        M = mMult*kappa*speye(m);
        C = [speye(m); -speye(m)];
        c = ones(2*m, 1);
        b = zeros(m,1);
        B = speye(m);     
        
    case 'l1'
        lam = params.lambda;
        M = 0*speye(m);
        C = [speye(m); -speye(m)];
        c = lam*ones(2*m, 1);
        b = zeros(m,1);
        B = speye(m); 
        
    case 'qreg' % lambda-scaled penalty.
        lam = params.lambda;
        tau = params.tau;
        M = 0*speye(m);
        C = [speye(m); -speye(m)];
        c = lam*[(1-tau)*ones(m,1); tau*ones(m,1)];
%        c = lam*ones(2*m, 1);
        b = zeros(m,1);
        B = speye(m);   

    case 'qhuber' % quantile huber penalty.
        tau = params.tau;
        kappa = params.kappa;
        M = kappa*speye(m);
        C = [speye(m); -speye(m)];
        c = [(1-tau)*ones(m,1); tau*ones(m,1)];
        b = zeros(m,1);
        B = speye(m);           
        
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

        
        
    case 'hinge'
        M = 0*speye(m);
        C = [speye(m); -speye(m)];
        c = [ones(m, 1); zeros(m,1)]; % hinge!
        b = zeros(m,1);
        B = speye(m);
              
    case 'l2m' % penalize everybody except the last element
        M = speye(m);
        C    = [zeros(2,m-1) ones(2,1)]; 
        c    = zeros(2,1);
        b    = zeros(m, 1);
        B    = speye(m);
        
    case 'l1m'
        lam = params.lambda;
        M = 0*speye(m);
        C = [speye(m); -speye(m); zeros(2,m-1) ones(2,1)];
        c = [lam*ones(2*m, 1); zeros(2,1)];
        b = zeros(m,1);
        B = speye(m); 

        
    otherwise
        error('unknown PLQ');
end

  % composing with linear model
        b = b-B*z;
        B = B*H;

end

