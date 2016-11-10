%% function to compute nonparametric sample Trimmed Expected Shortfall
% data: historical losses
% 0<alpha<beta<1, the quantiles that define the appropriate interval

function res=TES(data,alpha,beta)
qa=quantile(data,alpha);
qb=quantile(data,beta);
res=mean(data((data>=qa)&(data<=qb)));
end
