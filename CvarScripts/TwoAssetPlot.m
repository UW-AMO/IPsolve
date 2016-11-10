%% This is the test program to show the nonconxity of VaR and Trimmed CVaR

clear;

c=yahoo;

IBMPrice = fetch(c,'IBM','Adj Close','01/01/07','01/01/16','m');
IBMPrice=flip(IBMPrice);
IBMRet=IBMPrice(2:end)./IBMPrice(1:(end-1))-1;
IBMLoss=-IBMRet;
AAPLPrice = fetch(c,'AAPL','Adj Close','01/01/07','01/01/16','m');
AAPLPrice=flip(AAPLPrice);
AAPLRet=AAPLPrice(2:end)./AAPLPrice(1:(end-1))-1;
AAPLLoss=-AAPLRet;
lambdas=linspace(0,1);
VaRs=[];
CVaRs=[];
TrimmedCVaRs=[];
WinsorizedCVaRs=[];
alpha=0.9;  % The tail probability for VaR and CVaR
beta=0.99;   % The second tail probability for Trimmed CVaR

for iter=1:length(lambdas)
    lambda=lambdas(iter);
    ret=lambda.*IBMLoss+(1-lambda).*AAPLLoss;
    VaRs(iter)=quantile(ret,alpha);
    CVaRs(iter)=ES(ret,alpha);
    TrimmedCVaRs(iter)=TES(ret,alpha,beta);
    WinsorizedCVaRs(iter)=WES(ret,alpha,beta);
end

plot(lambdas,VaRs,'-+',lambdas,CVaRs,'-o',lambdas,TrimmedCVaRs,'-*');
legend('VaR','ES','RES');

