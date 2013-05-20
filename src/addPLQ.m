% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------




function [bout, cout, Cout] = addPLQ(b1, c1, C1, b2, c2, C2)



bout = [b1; b2];
cout = [c1; c2];

Cout = [C1, 0*speye(size(C1,1), size(C2,2));
    0*speye(size(C2,1), size(C1,2)) C2];

%Cout = Cout';

%Mout = [M1, 0*speye(size(M1,1), size(M2,2));
 %   0*speye(size(M2,1), size(M1,2)) M2];
 
 

end