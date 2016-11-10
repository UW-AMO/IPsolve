%% function to compute nonparametric sample Winsorized Expected Shortfall
% data: historical losses
% 0<alpha<beta<1, the quantiles that define the appropriate interval

function res=WES(data,alpha,beta)
qa=quantile(data,alpha);
qb=quantile(data,beta);
data(data>=qb)=qb;
res=mean(data(data>=qa));
end
