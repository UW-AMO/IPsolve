function [solution ] = PseudoSmoother(G, H, z, Q, R, n, m, N, x0, models)
Qps = blktridiag(pinv(Q), zeros(n,n), zeros(n,n), N);
Rps = blktridiag(pinv(R), zeros(m,m), zeros(m,m), N);

Gmain = blktridiag(speye(n,n), -G, zeros(n,n), N);
Hmain = blktridiag(H, zeros(m,n), zeros(m,n), N);

w = [x0; zeros((N-1)*n,1)];

params.K = sqrt(Qps)*Gmain;
params.k = sqrt(Qps)*w;

[solution, ~] = run_example(sqrt(Rps)*Hmain, sqrt(Rps)*z, models, 'l2', [], params);



end

