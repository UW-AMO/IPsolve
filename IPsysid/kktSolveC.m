function [ds, dq, du, dr, dw, dy] = kktSolve(a, A, b, B, c, C, M, s, q, u, r, w, y, mu)

r1      = -s - C'*u + c;
r2      = mu + (C'*u - c).*q;
r3      = -(B*y - M*u - C*q + b) + C*(r2./s);
T       = M + C*diag(q./s)*C';

if(~all(A))
    r4      = -(r + A'*y - a);
    r5      = mu + (A'*y - a).*w;
    r6      = -(B'*u + A*w) + B'*(T\r3) - A*(r5./r);
    Omega   = B'*(T\B) + A*diag(w./r)*A';
else
    r6      = -B'*u + B'*(T\r3);
    Omega   = B'*(T\B);
end

dy      = Omega\r6;

if(~all(A))
    dw      = (r5 + diag(w)*A'*dy)./r;
    dr      = r4 - A'*dy;
else
    dw = [];
    dr = [];
end
du      = T\(-r3 + B*dy);
dq      = (r2 + diag(q)*C'*du)./s;
ds      = r1 - C'*du;


end