function [ statesoln, vsoln, wsoln ] = extractor( m, n,N, soln )


%extracting states from soln
Dstateblock = [sparse(n,n) sparse(n,m) speye(n,n)];
Dstate = blktridiag(Dstateblock,sparse(n, 2*n+m), sparse(n, 2*n+m),N);
Dstate = [speye(n,n) sparse(n, N*(2*n +m)); sparse(N*n, n) Dstate];
statesoln = Dstate*soln;
%reshaping state solution to have desired form
statesoln = reshape(statesoln, n, N+1);

%extracting w's from soln (these are actually the u's in z in the writeup)
wblock = [speye(n,n), sparse(n,m), sparse(n,n)];
wextract = blktridiag(wblock, sparse(n, 2*n + m), sparse(n, 2*n+m),N);
wextract = [sparse(n*N, n), wextract];
wsoln = wextract*soln;
%reshaping
wsoln = reshape(wsoln,n, N);

%finally extracting v's from soln (these are the t's in z in the writeup)
vblock = [sparse(m,n), speye(m,m), sparse(m,n)];
vextract = blktridiag(vblock, sparse(m, 2*n+m), sparse(m, 2*n+m), N);
vextract = [sparse(m*N, n), vextract];
vsoln = vextract*soln;
%reshaping
vsoln = reshape(vsoln,m, N);



end

