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
zbar = z0;
ytemp = y0;
conv = 1;
iter = 0; 

[q,r] = qr(A',0);

AAt = A*A'; 
%qthatw = q'*hatw; 
%rrt = r*r';
t2 = lsqr(AAt, hatw, 1e-10, 1000);
At2 = A'*t2; 

while conv > 1e-5
    iter = iter +1;
    params.lambda = sigma;
    
    %updating y
    params.silent = 1;
    b = ytemp + sigma*D*zbar;
   
    [ynew] = run_example(speye(n), b, 'l2', 'l2', [], params);
    
    %updating z using QR + single what solution. 
    c = ztemp - tau*D'*ynew;
    t1 = c-q*(q'*c); % orthogonal project onto null(A). 
    znew = t1+At2;

    
    
%    znew = SingularSystemSolver(A, hatw, c,At2,q);
    zbar = 2*znew - ztemp;

    conv = norm(ztemp - znew)/tau + norm(ytemp-ynew)/sigma;

    
    fprintf('iter: %d, diff: %7.3e\n', iter, conv); 
    ztemp = znew;
    ytemp = ynew;
end

xopt = ztemp;

    
    

