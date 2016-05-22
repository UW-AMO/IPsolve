% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------

function [dq, du, dr, dw, dy, params] = kktSolveBarrier(linTerm, b, Bm, c, C, Mfun, q, u, r, w, y, params)


inexact = params.inexact;
simplex = params.simplex;
delta = params.delta;



%useChol = params.useChol;
pSparse = params.pSparse;
pFlag = params.pFlag;
pCon = params.constraints;
funM = isa(Mfun, 'function_handle');

mu = params.mu;
n = params.n;
%m = params.m;
m = size(Bm, 1);

% defining auxiliary terms
d = c - C'*u;
Q = spdiags(q, 0, length(q), length(q));
QD = spdiags(q./d, 0, length(q), length(q));

if(pCon)
    WR = spdiags(w./r, 0, length(w), length(w));
else
    WR = 0;
end

if(pFlag)
    MMn = params.M2;
    um = u(1:m);
    un = u(m+1:end);
else
    um = u;
end

if(funM)
    [Mum, MMm] = Mfun(um);
else
    MMm = Mfun;
    Mum = MMm*um;
end

if(pFlag)
    Mu = [Mum; MMn*un];
else
    Mu = Mum;
end

if(pFlag)
    MM = [MMm + params.rho*speye(size(MMm)), sparse(size(MMm,1), size(MMn,2));
        sparse(size(MMn,1), size(MMm,2)) MMn + params.rho*speye(size(MMn))];
else
    MM = MMm;
end


% need to improve this for infinity norm!!
% idea: implement Tinv instead of T. C'*C is sparse, but CC' is not. 
if(simplex)
    if(~pFlag)
        T1 = MM + C(:, 3:end)*QD(3:end,3:end)*C(:,3:end)';
        T1inv = myInvDiag(T1);
        T1invC = T1inv*C(:,1:2);
        % CT1inv = C(:,1:2)'*T1inv;
        Dinv = QD(1:2, 1:2)\speye(2) + C(:,1:2)'*T1invC;
        TinvF = @(x)T1inv*(x - (C(:,1:2)*(Dinv*(C(:,1:2)'*(T1inv*x)))));
    else % assumes simplex is on the regularizer (for now)
        Tm = MM(1:m,1:m) + C(1:m,:)*QD*C(1:m,:)';
        Tminv = myInvDiag(Tm);
        TminvF = @(x)Tminv*x; 
        T1 = MM(m+1:end, m+1:end) + C(m+1:end, 3:end)*QD(3:end,3:end)*C(m+1:end,3:end)';
        T1inv = myInvDiag(T1);
        T1invC = T1inv*C(m+1:end,1:2);
        Dinv = myInvDiag(myInvDiag(QD(1:2, 1:2)) + C(m+1:end,1:2)'*T1invC);
        TninvF = @(x)T1inv*(x - (C(m+1:end,1:2)*(Dinv*(T1invC'*x))));
        TinvF = @(x)funcStack(x, TminvF, TninvF, m); %funcstack!
%        TinvF = @(x)[ TminvF*x(1:m,:); TninvF(x(m+1:end),:)];
    end
else
    T       = MM + C*QD*C';
    Tinv    = T\speye(size(T));
    TinvF   = @(x)Tinv*x; 
end

%Torig       = MM + C*QD*C';


    
% if two pieces, exploit structure
if(pFlag)
     Bn = params.B2;
     if(~simplex)
         Tm = T(1:m,1:m);
         Tminv = Tinv(1:m, 1:m);
         TminvF = @(x)Tminv*x;
         Tn = T(m+1:end, m+1:end);
         Tninv = Tinv(m+1:end, m+1:end);
         TninvF = @(x)Tninv*x;
     end
end



% here we go!
r1 = -d.*q + mu;

% r2 changed a bit
if(pFlag)
    r2      = -([Bm*y; Bn*y] - Mu + b) + mu*C*(1./d); % removed -Cq from inside parens
else
    r2      = -(Bm*y - Mu + b) + mu*C*(1./d); % removed -Cq from inside parens
end


if(pCon)
    A = params.A;
    a = params.a;
    r3 = -r.*w + mu;
    r4 = -A'*y + a - mu*(1./w);
    Awr4r = A*(w - WR*r4);
%    SpOmegaMod =  A*WR*A';
else
    A = 0;
    Awr4r = 0;
%    SpOmegaMod = 0*speye(n);
end

utr = u - TinvF(r2); % should be defined regardless

if(pFlag)
    r5      = -Bm'*utr(1:m) - Bn'*utr(m+1:end) - Awr4r -linTerm;
else
    r5      = -Bm'*utr - Awr4r - linTerm;
end


% compute dy
if pFlag && n > m && pSparse &&~inexact
    %    BTB = Bn'*(Tn\Bn) + SpOmegaMod; % large sparse matrix
    if(simplex)
%        @(x)T1inv*(x - (C(m+1:end,1:2)*(Dinv*(T1invC'*x))));
       BnT1inv = Bn'*T1inv; 
        BTB = BnT1inv*Bn - (BnT1inv*C(m+1:end,1:2))*(Dinv*(T1invC'*Bn))+ A*WR*A'+ delta*speye(size(Bn,2));
 %       BTB = Bn'*(TninvF(Bn)) +A*WR*A' + delta*speye(size(Bn,2));
    else
        BTB = Bn'*(TninvF(Bn)) +A*WR*A' + delta*speye(size(Bn,2));
    end
    
    Air4 = BTB\r5;
    res = Bm*Air4;
    
    % indirect (working) version
    if(inexact)
        
        TBAB = @(x) Tm*x + Bm*(BTB\(Bm'*x)) + x*params.delta;
        
        %maxElt = max(diag(BTB));
        %precon = diag(max(maxElt*ones(m,1), diag(Tm)));
        [u, FLAG,RELRES,ITER] = pcg(TBAB, res, params.tolqual, 10000);
        params.info.pcgIter = [params.info.pcgIter ITER];
        
        % direct working version
    else
        TBAB = Tm + Bm*(BTB\Bm');
        u = TBAB\res;
    end
    
    dy = Air4 - BTB\(Bm'*u);
    
    
else
    if(pFlag)
        if(inexact)
            if(pCon)
                if(simplex)
                      BnT1inv = Bn'*T1inv; 
                      BTB = BnT1inv*Bn - (BnT1inv*C(m+1:end,1:2))*(Dinv*(T1invC'*Bn))+ delta*speye(size(Bn,2));
                    
                else
                    BTB = Bn'*(TninvF(Bn)) + A*WR*A' + delta*speye(size(Bn,2));
                end
            else
               % tic
               if(simplex)
                      BnT1inv = Bn'*T1inv; 
                      BTB = BnT1inv*Bn - (BnT1inv*C(m+1:end,1:2))*(Dinv*(T1invC'*Bn))+ delta*speye(size(Bn,2));

               else
                BTB = Bn'*(TninvF(Bn)) + delta*speye(size(Bn,2));
               end
                %toc
            end
            Omega   =  @(x) BTB*x + Bm'*(TminvF((Bm*x))) + params.delta*x;
 %           dTn = diag(Tn);
%            elt = max(max(dTn(1:m), diag(Tm)));
%            eltTwo = max(max(diag(Tn)), max(diag(Tm)));
%            precon = (1/eltTwo)*(Bn)'*Bn;
            
%            mat1 = (Tm + (params.Anorm^2/delta)*speye(size(Tm)));
 %           mat2 = Tn + 1*speye(size(Tn)); 
%            mat = blkdiag((Tm + (params.Anorm^2/delta)*speye(size(Tm))), Tn + 1*speye(size(Tn)));
            
%            mat3 = BTB + speye(size(BTB))/delta;

%            preCon = @(x) mat3\x - Bm'*(mat1\(Bm*x))/delta^2- Bn'*(mat2\(Bn*x))/delta^2;
%            preCon = @(x) x/delta - [Bm' Bn']*(mat\([Bm*x; Bn*x]))/delta^2;


            
%            preCon = @(x) x/delta - Bm'*((Tm + (params.Anorm^2/delta)*speye(size(Tm)))\(Bm*x))/delta^2;
%            [dy] = pcg(Omega, r6, params.tolqual, 10000, preCon);

            [dy, FLAG,RELRES,ITER] = pcg(Omega, r5, params.tolqual, 10000);
            params.info.pcgIter = [params.info.pcgIter ITER];

            
            

%            [dy] = minres(Omega, r6, params.tolqual, 10000);
            
            
%            preCon = @(x) x/delta - Bm'*(Tm\(Bm*x));

            
        else
            Omega   =  Bn'*(TninvF(Bn))+ Bm'*(TminvF(Bm)) +delta*speye(size(Bm,2));
            if(params.constraints)
                Omega = Omega + A*WR*A';
            end
            OmegaChol = chol(Omega);
            dy      = OmegaChol\(OmegaChol'\r5);
        end
    else
        if(inexact)
            if(pCon)
                
                Omega   =  @(x) A*(WR*(A'*x)) + Bm'*(TinvF(Bm*x)) + delta*x;
            else
                Omega   =  @(x) Bm'*(TinvF(Bm*x)) + delta*x;
            end

      %      preCon = @(x) x/delta - Bm'*((T + (params.Anorm^2/delta)*speye(size(T)))\(Bm*x))/delta^2;
      %      [dy] = pcg(Omega, r6, params.tolqual, 10000, preCon);
            
            
            [dy, FLAG,RELRES,ITER] = pcg(Omega, r5, params.tolqual, 10000); 
            params.info.pcgIter = [params.info.pcgIter ITER];

            
        else
            if(pCon)
                Omega   = Bm'*(TinvF(Bm)) + A*WR*A' + delta*speye(size(Bm,2));
            else
                Omega   = Bm'*(TinvF(Bm)) + delta*speye(size(Bm,2));
            end
%            dy = Omega\r5;
            OmegaChol = chol(Omega);
            dy      = OmegaChol\(OmegaChol'\r5);
        end
    end
    
    %dy = Omega\r6;
    %    [dy, ~] = lsqr(Omega, r6, [], 5);
end

% done computing dy


% compute dw and dr
if(pCon)
    dw = -WR*(r4 - A'*dy);
    dr = (r3-r.*dw)./w;
%   dw      = (r5 + W*(params.A'*dy))./r;
%    dr      = r4 - params.A'*dy;
else
    dw = [];
    dr = [];
end


% compute du
if(pFlag)
    du      = TinvF(-r2 + [Bm*dy; Bn*dy]);
else
    du      = TinvF(-r2 + Bm*dy);
end


%compute dq 
dq      = (r1 + Q*C'*du)./d;

end