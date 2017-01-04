function [Lagsoln, Relaxsoln, Robustsoln ] = SingularKalmanRunner(G, H, Q, R, x_0, N, z, lambda, othersoln)
%SingularKalmanRunner
    %INPUTS:
        %This takes information about a model of the form
            %x_k+1 = G_k x_k + w_k
            %z_k = H_k x_k + v_k
            %Where Q, R are the covariances of the state and measurment
            %respectivly
        %and solves for estimates of the states
        %
        %For now we are assuming that G, H, Q, R do not depend upon k
        %
        %x_0 is the guess of the initial state (a column vector) and all others should be
        %matrices of the appropriate size
        %
        %N is the total number of measurments (do not count x_0)
        %
        %z is the observed measurments. They should be entered as a single
        %column vector with all measurments concatenated together.
        %
        %lambda is the parameter to be used in the relaxed and robust
        %solver
        %
        %othersoln specifes whether the output solution is state soultions
        %or w or v soultions. To specify enter 'state', w' or 'v'. The
        %default value is 'state'
    %OUTPUTS:
        %All solutions (state, w, and v) to Lagrangian, Relaxed, and Robust
        %solver are given. All are given as matrices with the solutions
        %given as columns. (note that statesoln has an initial state while
        %w and v do not)

        
if ~exist('othersoln', 'var')
    othersoln = 'state';
end

[D, A, hatw] = SingularKalmanSetup(G, H, Q, R, x_0, N, z);

n = length(x_0);
m = length(z)/N;

[Lagsoln, Lagtime] = LagrangianSolver(A, D, hatw, n, m, N, othersoln);

[Relaxsoln, Relaxtime] = RelaxedSingularSolver(A, D, hatw, n, m, N, lambda^2, othersoln);

[Robustsoln, Robusttime] = RobustSingularSolver(A, D, hatw, n, m, N, lambda, othersoln);

fprintf('size: %d, time Lagrangian: %7.2e, time relaxed: %7.2e, time robust: %7.2e', N*n, Lagtime, Relaxtime, Robusttime);
    


end

