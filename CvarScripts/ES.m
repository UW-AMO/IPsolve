%% Function to compute Expected Shortfall directly from loss vector

function res=ES(data,alpha)
qa=quantile(data,alpha);
res=mean(data(data>=qa));
end