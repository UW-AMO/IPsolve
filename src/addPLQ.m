% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------




function [bout, cout, Cout] = addPLQ(b1, c1, C1, b2, c2, C2)


bout = [b1; b2];
cout = [c1; c2];

Cout = blkdiag(C1,C2);
% if(nnz(C1) == 0)
%     Cout = blkdiag(sparse(size(C1,1)), C2);
% elseif(nnz(C2) == 0)
%     Cout = blkdiag(C1, sparse(size(C2,1)));
% else    
%     [C1, sparse(size(C1,1), size(C2,2));
%         sparse(size(C2,1), size(C1,2)) C2];
% end
   
%Cout = Cout';

%Mout = [M1, 0*speye(size(M1,1), size(M2,2));
 %   0*speye(size(M2,1), size(M1,2)) M2];
 
 

end