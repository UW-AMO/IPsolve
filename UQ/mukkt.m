function [ r1, r2, r3 ] = mukkt( s, lam, x, mu, params )

r1 = params.C*x + s - params.d;
r2 = s.*lam - mu;
r3 = params.A'*(params.A*x - params.b) + params.C'*lam;


end

