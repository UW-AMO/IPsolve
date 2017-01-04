function [ D, A, hatw ] = SingularKalmanSetup(G, H, Q, R, x_0, N, z )%
%SingularKalmanSetup
    %This function takes information about the Kalman model being used and
    %creates new notation that will be used in the optimization problem.
    
    %Here the Kalman model is assumed to have the form:
        %x_k+1 = G_k x_k + w_k
        %z_k = H_k x_k + v_k
        %Where Q, R are the covariances of the state and measurment
        %respectivly
        
    %For now we are assuming that G, H, Q, R do not depend upon k
    
    %x_0 is the guess of the initial state (a column vector) and all others should be
    %matrices of the appropriate size
    %N is the total number of measurments (do not count x_0)
    %z is the observed measurments. They should be entered as a single
    %column vector with all measurments concatenated together.

n = length(x_0);

m =length(z)/N;

%taking square roots
multMatSing = sqrt(Q);
Rmat = sqrt(R);

%making matrix A
Diagblock = [ sparse(m, n), sparse(m,m), sparse(m,n);  multMatSing, sparse(n,m), speye(n)];
UDblock = [sparse(m, n), Rmat, H; sparse(n,n), sparse(n,m), -G  ]; 
[dim1, dim2] = size(UDblock); 
Ourmat = blktridiag(Diagblock,UDblock, sparse(dim1,dim2),N);


[rownum, colnum] = size(Ourmat); 
newCol = sparse(rownum, n); 
newCol(m+1:m+n, 1:n) = -G;
Ourmat = [newCol, Ourmat];
[rownum, colnum] = size(Ourmat); 
newRow = sparse(n, colnum); 
newRow(1:n, 1:n) = speye(n); 
newRow = [newRow];
Ourmat = [newRow; Ourmat]; 
anotherrow = [sparse(m,n+(N-1)*(2*n+m)) sparse(m,n) Rmat H];
Ourmat = [Ourmat; anotherrow];

A = Ourmat;

%Making D matrix
Dblock = blkdiag(sparse(n,n), speye(n,n), speye(m,m));
blockdim = 2*n + m;
D = blktridiag(Dblock, sparse(blockdim, blockdim), sparse(blockdim, blockdim),N);
D = [D sparse(N*(2*n + m),n)];
D = vertcat(D, sparse(n,N*(2*n + m) + n));

%Forming hatw
zmat = reshape(z, m,N);
zz = [sparse(n,N); zmat];
hatw = zz(:);
w0 = x_0;
hatw = [w0; sparse(m,1);hatw];

    

end

