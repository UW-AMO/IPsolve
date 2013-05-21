function [ g, H, f] = studentFunc( u, m )
%QUADFUNC Summary of this function goes here
%   Detailed explanation goes here

scaledU = u/m;


omu2 = 1-scaledU.^2;  % vector of weights

if(omu2 <= 0)
    error('omu2 <= 0'); 
end


f = 1-log(2)-sqrt(1-scaledU.^2) + log(1+sqrt(omu2)); 

g = (1/m)*(scaledU./sqrt(1-scaledU.^2) + scaledU./(sqrt(omu2)) - scaledU./(omu2 + sqrt(omu2)));

if(nargout > 1)
   v = (1/m)*(1./(1-scaledU.^2).^(3/2) - (1+(1+scaledU.^2).*sqrt(omu2))./(sqrt(omu2).*((sqrt(omu2)+omu2).^2))); 
   H = 0*speye(length(u));
   if(length(u) == 1)
       H = v;
   else
       H = spdiags(v, 0, H);
   end
end


end

