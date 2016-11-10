%% function to compute the CVaR and VaR of the given losses for the given tail probabilities
function [myCVaR, myVaR]=myCVaR(data,prob,eq_tol=1e-10)
m=length(data);
rsort=sort(data);
quantile_loc=ceil(m*prob);
myVaR=rsort(quantile_loc);
myCVaR=mean(rsort(rsort>=(myVaR-eq_tol)));
end

