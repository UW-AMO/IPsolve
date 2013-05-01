function [ M C c b B ] = loadPenalty( H, z, penalty, params )
%loadPENALTY Loads one of several predefined penalties, 
% for IP solver. 

m = params.size;

switch(penalty)
    case 'huber'
        M = speye(m);
        C = [speye(m); -speye(m)];
        c = params.kappa*ones(2*m, 1);
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

        
    case 'l2'
        M = speye(m);
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
        
    otherwise
        error('unknown PLQ');
end

  % composing with linear model
        b = b-B*z;
        B = B*H;

end

