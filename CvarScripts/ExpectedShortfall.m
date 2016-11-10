
%% function to compute expected shortfall

% x is the weights
% y is the loss matrix
% beta is the tail probability

function res=ExpectedShortfall(x,y,beta)
m=size(y,1);
quantile_loc=ceil(beta*m);
r=y*x;
rsort=sort(r);
q=rsort(quantile_loc);
res=mean(r(r>=q));
end

