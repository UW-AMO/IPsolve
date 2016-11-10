function[ xopt ] = CPSolver(D, A, hatw, z0, y0)
%This function solves the problem
% 1/2||Dz||^2 s.t. Az=hatw
%using a Chambolle-Pock algorithm
%Both D and A are allowed to be singular but (for now) we assume hatw is in
%the range of A
%
%z0 and y0 are the initial points for the primal and dual variables
%respectivly
%We will use the stopping critererion that |zn-zn+1|<.01

%set parameters
n = length(z0);
sigma = .5;
tau = .5;
ztemp = z0;
ytemp = y0;
difference = 1;
while difference > .01
    b = ytemp + sigma*D*ztemp;
    params.lambda = sigma;
    
    %updating y
    [opt, ynew] = run_example(speye(n), b, 'l2', 'l2', [], params);
    
    %updating z
    c = ztemp - tau*D'*ynew;
    znew = SingularSystemSolver(A, hatw, c);
    
    difference = abs(ztemp - znew);
    
    ztemp = znew;
    ytemp = ynew;
end

xopt = ztemp;

    
    

