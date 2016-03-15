% -----------------------------------------------------------------------------
% ipSolve: General Interior Point Solver for PLQ functions Copyright (C) 2013
% Authors: Aleksandr Y. Aravkin: sasha.aravkin at gmail dot com
% License: Eclipse Public License version 1.0
% -----------------------------------------------------------------------------




function [bout, Bout, cout, Cout, Mout] = addPLQfull(b1, B1, c1, C1, M1, b2, B2, c2, C2, M2)



bout = [b1; b2];
Bout = [B1; B2];

cout = [c1; c2];

Cout = blkdiag(C1, C2); 

Mout = blkdiag(M1,M2); 
 
 

end