format long
%

dbstop if error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  The problem we are trying to solve is
%
% min ||u||_1 s.t. ||F(u)||_1 = 0
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Initialize
%
params.debg=0;
debg=params.debg;

params.mu_rate =10;
params.mu_max = 100.0;
params.epsilon_Si = 1.0e-7;     % Tolerances in Si
params.epsilon_Sa = 1.0e-7;     % Initial tolerances in Sa
params.eps_stationary    = 1.0e-7;% Initial first-order tolerance
params.eps_stop = 1.0e-5;      % Stopping epsilon for the first order conditions
params.epsilon = 1.0e-9;       % other tolerances

t_start = tic;
    

for ii= 9:9
    
    outer_iter=0;
    params.mu = 10;
    pNum       = ii;         % Simple test problem
    params.test = pNum;      % in case it is used within code
    params = getProblemLinear(pNum, params);
    %Added noise
    % min ||u||_1 s.t. ||F(u)||_1 = 0
    
    %
    % initial point
    %
    mu_max = params.mu_max;
    mu = params.mu;
    
    
    %
    % Set up diary
    %
    if debg
        data=strcat('../Diary/junk_',int2str(params.mu),'_',int2str(pNum),'.txt');
    else
        data=strcat('../Diary/4linear_l1_test_',int2str(params.mu),'_',int2str(pNum),'.txt');
    end
%    diary(data)    
    %
    x_0 = params.x0;
    [x, inner_iter, outer_iter, fevals, lambda, g, F, Si, Sa, d, shift, optimal,params] = outer(x_0, outer_iter, params);
    %
    n=length(x);
    fprintf('\n\nInitial value for mu of = %4d \n\n',mu);
    mu = params.mu;
    fprintf('\n\nInitial l_1 norm of x  = %4d \n\n', norm(x_0,1))
    [F,fval]=lsF(F,x,0,d,0,shift,params);
    if shift == 0
        fprintf('Current best l_1 Penalty value = %15.13e \n\n', fval)
        fprintf('x = %13.8e %13.8e %13.8e \n\n', x)
    else
        fprintf('Current best PERTURBED l_1 Penalty value = %15.13e \n\n', fval)
        fprintf('PERTURBED x = %13.8e %13.8e %13.8e \n\n', x)
    end
    if optimal
        if mu > mu_max
            fprintf('\n\nThreshold for mu of = %4d  exceeded \n\n',mu_max);
        else
            fprintf('\n\nFinal value for mu of = %4d \n\n',mu);
        end
        fprintf('Multipliers are:\n\n');
        for i=1:length(Si)+length(Sa)
            fprintf('%4d\n', lambda(i));
        end
        fprintf('\n\nSolution:\n\n');
        for i=1:n
            fprintf('%4d\n ', x(i));
        end
        fprintf('\n\nl_1 norm of x  = %4d \n\n', norm(x,1))
        fprintf('l_1 norm of F  = %4d \n\n',norm(F,1));
        m = length(F);
        fprintf('Final values of F are:\n\n');
        for i=1:m
            fprintf('%4d\n ', F(i));
        end
    else
        fprintf('\n\nOptimal Solution not necessarily found : l_1 norm of x  = %4d \n\n', norm(x,1));
        fprintf('l_1 norm of F  = %4d \n\n',norm(F,1));
        if mu > mu_max
            fprintf('\n\nThreshold for mu of = %4d  exceeded \n\n',mu_max);
        else
            fprintf('\n\nFinal value for mu of = %4d \n\n',mu);
        end
        fprintf('Multipliers are:\n\n');
        for i=1:length(Si)+length(Sa)
            fprintf('%4d   ', lambda(i));
        end
        fprintf('\n\nSolution:\n\n');
        for i=1:n
            fprintf('%4d   ', x(i));
        end
        fprintf('\n\nl_1 norm of x  = %4d \n\n', norm(x,1))
        fprintf('l_1 norm of F  = %4d \n\n',norm(F,1));
        m = length(F);
        fprintf('Final values of F are:\n\n');
        for i=1:m
            fprintf('%4d   ', F(i));
        end
    end
    fprintf('\n\nTotal number of outer iterations  = %4d \n\n', outer_iter);
    fprintf('Total number of inner iterations  = %4d \n\n', inner_iter);
    fprintf('Total number of function evaluations  = %4d \n\n', fevals);
    toc(t_start);
    diary off
end
%
