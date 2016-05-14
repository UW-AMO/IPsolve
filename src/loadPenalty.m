% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [ M, C, c, b, B, fun, prox_g] = loadPenalty( H, z, penalty, params )
%loadPENALTY Loads one of several predefined penalties,
% for IP solver.

m = params.size;
prox_splx = @(y) max(y-max((cumsum(sort(y,1,'descend'),1)-1)./(1:size(y,1))'),0);
%box_proj_general = @(y,high,low)max(min(y, high), low);


switch(penalty)
    
    case'infnorm'

        M = 0*speye(2*m);
        lam = params.lambda;
        C = [ ones(1, 2*m); -speye(2*m)]; % sum(u) < = 1, don't need the other! 
        c = [lam; zeros(2*m,1)];
        B = [speye(m); -speye(m)];
        b = zeros(2*m,1);
        fun = @(x) lam*norm(x,inf);        
        prox_g = @(y,gamma) lam*prox_splx(y/lam);
         
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
        prox_g = @(y,gamma)box_proj_general(y,lam, 0);
        
        
    case 'huber'
        kappa = params.kappa;
        mMult = params.mMult;
        M = mMult*kappa*speye(m);
        C = [speye(m); -speye(m)];
        c = ones(2*m, 1);
        b = zeros(m,1);
        B = speye(m);
        
        % function handle to evaluate objective
        fun = @(x) sum((abs(x) > mMult*kappa).*(abs(x) - 0.5*mMult*kappa) + 0.5*(abs(x) <= mMult*kappa).*x.^2/(mMult*kappa));
        prox_g = @(y,gamma)box_proj_general(y, 1,-1);

        % function handle for gradients, in case this is needed elsewhere
        params.gfun = @(x) (abs(x) > mMult*kappa).*sign(x) + (abs(x) <= 0.5*mMult*kappa).*x/(mMult*kappa);
        
    case 'l1'
        lam = params.lambda;
        M = 0*speye(m);
        C = [speye(m); -speye(m)];
        c = lam*ones(2*m, 1);
        b = zeros(m,1);
        B = speye(m);
        
        % function handle to evaluate objective
        fun = @(x) lam*norm(x,1);
        prox_g = @(y,gamma)box_proj_general(y, lam, -lam);
        
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
        prox_g = @(y,gamma)box_proj_general(y, (1-tau)*lam, -lam*tau);
        
        
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
        prox_g = @(y,gamma)box_proj_general(y, (1-tau)*2, -2*tau);
        %sum(2*tau*abs(x(left)) - 2*thresh*tau^2)
        
    case 'l2func'
        mMult    = params.mMult;
        M    = @(x)quadFunc(x, mMult*speye(m));
        C    = zeros(1,m); % easy to satisfy 0'*u <= 1
        c    = 1;
        b    = zeros(m, 1);
        B    = speye(m);
        prox_g = @(y,gamma) y;
        
    case 'l2'
        mMult    = params.mMult;
        M    = mMult*speye(m);
        C    = zeros(1,m); % easy to satisfy 0'*u <= 1
        c    = 1;
        b    = zeros(m, 1);
        B    = speye(m);
        
        % function handle to evaluate objective
        fun = @(x) norm(x)^2/(2*mMult);
        prox_g = @(y,gamma) y;

        
    case 'hinge'
        M = 0*speye(m);
        lam = params.lambda;
        C = [speye(m); -speye(m)];
        c = lam*[ones(m, 1); zeros(m,1)]; % hinge!
        b = zeros(m,1);
        B = speye(m);
        
        % function handle to evaluate objective
        fun = @(x) lam*sum(pos(x));
        prox_g = @(y,gamma)box_proj_general(y, lam, 0);
  
        
        
case 'l2m' % penalize everybody except the last element
        mMult = params.mMult; 
        M = mMult*speye(m);
        C    = [zeros(2,m-1) [1; -1]]; % to make u<=0, -u <= 0.
        c    = 1e-4*ones(2,1);
        b    = zeros(m, 1);
        B    = speye(m);
        
        % function handle to evaluate objective
        fun = @(x) 0.5*norm(x(1:end-1))^2;

        prox_g = @(y,gamma) [y(1:end-1); 0]; % WTF
  
        
    case 'l1m'
        lam = params.lambda;
        M = 0*speye(m);
        C = [speye(m); -speye(m); zeros(2,m-1) [1; -1]];
        c = [lam*ones(2*m, 1); zeros(2,1)];
        b = zeros(m,1);
        B = speye(m);
        
        % function handle to evaluate objective
        fun = @(x) lam*norm(x(1:end-1), 1);
         prox_g = @(y,gamma) [box_proj_general(y(1:end-1), lam, -lam); 0];  % WTF
        
        
    case 'logreg'
        %        scale    = params.scale;
          alpha = 0.5;
        epsil = 0; 
        M    = @(x)logRegFunc(x,alpha);
        %        M    = opFunction(m, m, M);
        %C    = zeros(1,m); % easy to satisfy 0'*u <= 1
        %c    = 1;
        C = [speye(m); -speye(m)];
        c = [ones(m, 1); epsil*ones(m,1)]; %
        b = zeros(m,1);
        B = -speye(m);
        fun = @(x) sum(log(1+exp(x)));
        prox_g = @(y,gamma)box_proj_general(y, 1-alpha, alpha); 
        
    case 'hybrid'
        scale    = params.scale;
        M    = @(x)hybridFunc(x, scale);
        %         C    = zeros(1,m); % easy to satisfy 0'*u <= 1
        %         c    = 1;
        C = [speye(m); -speye(m)];
        c = scale^2*ones(2*m, 1); %
        
        b    = zeros(m, 1);
        B    = speye(m);
        
        % function handle to evaluate objective
        fun = @(x) sum(sqrt(1 + x.^2/scale) - 1);
        prox_g = @(y,gamma)box_proj_general(y, scale^2, -scale^2); 
        % function handle for gradients, in case this is needed elsewhere
        %        params.gfun = @(x) x./sqrt(1 + x.^2/scale);
        %%
%     case 'studentPL'
%         scale    = params.scale;
%         %        kappa = params.kappa;
%         C = [speye(m); -speye(m)];
%         c = scale^2*ones(2*m, 1);
%         M    = @(x)studentFunc(x, scale);
%         b    = zeros(m, 1);
%         B    = speye(m);
%         fun  = @(x)sum(log(1+((x./scale).^2)));
%         %        fun = @(x)sum(1-log(2)-sqrt(1-(x./scale).^2) + log(1+sqrt(1-(x./scale).^2)));
%         
%         
%     case 'student'
%         scale    = params.scale;
%         M    = @(x)studentFunc(x, scale);
%         C    = zeros(1,m); % easy to satisfy 0'*u <= 1
%         c    = 1;
%         b    = zeros(m, 1);
%         B    = speye(m);
%         
    otherwise
        error('unknown PLQ');
end



% composing with linear model
%b = b-B*z;
b = -b-B*z; 
%B = B*H;
B = B*H;

end

