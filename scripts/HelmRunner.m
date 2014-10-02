            seed_feas     = 1;      %seed to generate the feasible point 
            sparsity_feas = 0.995;   %level of sparsity of the feasible point 
            seed_init     = 2;      %seed to generate the finitial point 
            sparsity_init = 0;      %level of sparsity of the initial point 
            % 
            %------------------------------------------------------------ 
            % define parameters 
            params.explicit = true; % explicit or implicit ocnstruction of the Helmholtz operator 
            params.fwd.direct = false;% direct vs. iterative solution 
            params.fwd.tol = 1e-6; % forward solution relative residual error tolerance 
            params.fwd.restart = 20; % restart for forward GMRES solver 
            params.fwd.maxit = 300; % forward itertive solver maximum number of iterations 
            params.fwd.prec = true; % flag for preconditioning 
            params.fwd.precType = 'sgs'; % preconditioning options, iLU, SGS 
            params.fwd.droptol = 1e-2; % drop tolerance for 
            params.verbose = false; % display information through the run 
            params.both = false; % sensitivities w.r.t. both alpha and kappa, otherwise, only kappa 
            params.displaySVDs = true; % display SVDs of the Jacobian 
            params.sparse = true; % kappa is ssparse 
            %------------------------------------------------------------ 
  
            % MODEL DIMENSIONS 
%             n1 = 16;    % grid size1 
%             n2 = 16;    % grid size2 
%             n3 = 4;    % grid size3 
  
            n = [8 8 4]; 
            %n = [16 16 4]; 
            params.n = n; 
            L = 2 * n; 
            
            %------------------------------------------------------------ 
            % PROBLEM DATA 
            kappa = ones(n); 
            %             kappa(4:12, 6:10, 2:6) = 100; 
            %             kappa(11:14, 3:9, 4:8) = 1000; 
            
            rho = 5e5 * ones(n); 
            %             rho(10:20, 10:20, 2:6) = 500; 
            %             rho(15:25, 30:45, 4:8) = 50; 
            omega = 5; 
            
            params.rho = rho; 
            params.omega = omega; 
                        
            
            %------------------------------------------------------------ 
            % get differential operators 
            [G, Acf1, Acf2, Acf3, params] = getDifferentialOperators(L, params); 
            
            %------------------------------------------------------------ 
            % generate dictionary 
            ocf = 10; % overcompleteness factor 
            D = getDCTdictionary(n, ocf); 
            params.D = D; 
            
            %------------------------------------------------------------ 
           % define sources and recivers 
           nq  = 16;
           % nq  = 32; 
            params.nq = nq; 
            q = setSources(n(1)-1,n(2)-1,n(3)-1,nq); 
            params.q = q; 
            
            %------------------------------------------------------------ 
            % define measurement operator 
                         nQ  = 16; 
            %nQ  = 32; 
            params.nQ = nQ; 
            Q = setSources(n(1)-1,n(2)-1,n(3)-1,nQ); 
            params.Q = Q; 
              
            %------------------------------------------------------------ 
            %FEASIBLE SOLUTION 
            rand ('state', seed_feas); 
            %U = abs(sign(sprand(n1*n2*n3*ocf/2,1,0.01))); 
            U = rand(prod(n)*ocf/2,1); 
            U (U < sparsity_feas) = 0; 
            
            %------------------------------------------------------------ 
            % compute forward and get data 
            F_0 = vec(Q' * solveHelmholtz(Helmholtz3DOperator(U, rho, omega, params), q, params)); 
            params.F_0 = F_0; 
            
            %------------------------------------------------------------ 
            %INITIAL POINT 
            rand ('state', seed_init); 
            %u0 = abs(sign(sprand(n1*n2*n3*ocf/2,1,0.5))); 
            u0 = rand(prod(n)*ocf/2,1); 
            u0(u0 < sparsity_init) = 0; 
            
            %x_0 = u0;
            x_0 = U;
            %initial_mu = 1e-4; 
            initial_mu = 1.0e0; 
            tol = 0;   
            
            [F J] = HelmFun(x_0, params);
            
            a = 5;
            