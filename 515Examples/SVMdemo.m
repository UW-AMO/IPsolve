clear all

%% generate data
nInst = 2000;
nVars = 200;
nTest =2000; 


X = randn(nInst,nVars);
wTrue = randn(nVars,1);
y = sign(X*wTrue + randn(nInst,1));

corrFunc = @(X,y,w) 100*sum(y.*sign(X*w) >0)/length(y); 


%% run IPsolve
yA = sparse(1:nInst, 1:nInst, y)*X; % pre-multiplied data
params.proc_mMult = 1e1; % small least squares regularization
%params.proc_lambda = 5e1; % 1-norm regularization

wSVM2 =  run_example( yA, ones(nInst,1), 'hinge', 'l2', [], params );

wSVM1 =  run_example( yA, ones(nInst,1), 'hinge', 'l1', [], params );


%% testing


Xtest = randn(nInst,nVars);
yTrue = sign(Xtest*wTrue);

corrSVM2 = corrFunc(Xtest, yTrue, wSVM2);

corrSVM1 = corrFunc(Xtest, yTrue, wSVM1);
fprintf('SVM-L2: %7.1f, SVM-L1: %7.1f\n', corrSVM2, corrSVM1);

%% LR tests (experimental)
% corrLR2  = corrFunc(Xtest, yTrue, wLR2);
% 
% corrLR1  = corrFunc(Xtest, yTrue, wLR1);

% params.uIn = 0.5; % start in middle of box
% wLR2  =  run_example( yA, 0*ones(nInst,1), 'logreg', 'l2', [], params );
% 
% wLR1  =  run_example( yA, 0*ones(nInst,1), 'logreg', 'l1', [], params );

% fprintf('SVM-L2: %7.1f, SVM-L1: %7.1f, LR2: %7.1f, LR1: %7.1f\n', ...
%                             corrSVM2, corrSVM1, corrLR2, corrLR1);


