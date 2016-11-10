clear;

% rng(2);
Y=randn(3,2);
beta=0.8;

%% contour of ES
figure(1)
ezcontour(@(x,y) ExpectedShortfall([x;y],Y,beta),[0,1],[0,1])


%% contour of VaR
figure(2)
ezcontour(@(x,y) quantile(Y*[x;y],beta))