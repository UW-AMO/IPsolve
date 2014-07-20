%% Load data
%close all
%clear all
%clc

global THETA_LBFGS;

fprintf('Loading Data\n');
load('rcv1_train.binary.mat'); data_flag = 1; dataset_name = 'rcv1';
%load('adult.mat'); data_flag = 2; dataset_name = 'adult';
%load('sido.mat'); data_flag = 3; dataset_name = 'sido';
X = [ones(size(X,1),1) X];
[size_dataset,p] = size(X);

perm_index = randperm(size(X,1));
X_train = X(perm_index(1:ceil(0.9*size(X,1))),:);
y_train = y(perm_index(1:ceil(0.9*size(X,1)))); 
X_test = X(perm_index(ceil(0.9*size(X,1))+1:end),:);
y_test = y(perm_index(ceil(0.9*size(X,1))+1:end));

%load('data_debug.mat');
load('bupa.mat');
data_flag = 1; 
dataset_name = 'bupa';

X = X_train;
y = y_train;
Xt = X';

Y = speye(length(y));

Y = spdiags(y,0,Y);

Xlab = Y*X;
Xlabt = Xlab';
[n,p] = size(X);

%% Set up problem
maxNOfPasses = 25; % 25 passes through the data set
maxIter = n*maxNOfPasses;
lambda = 1/n;
yOld = y;
%objective = @(w)(1/n)*LogisticLoss(w,X,y) + (lambda/2)*(w'*w);
objective = @(w)LogisticReg(w,X,y, lambda, n);
objective_test = @(w)(1/n)*LogisticLoss(w,X_test,y_test);

% Order of examples to process
iVals = int32(ceil(n*rand(maxIter,1)));

fractionPass = 10;  %every 1/fractionPass of the pass is plotted

% %%
% fprintf('Running LBFGS\n');
% 
% w = zeros(p,1);
% params.X = X;
% params.y = y;
% params.lambda = lambda;
% THETA_LBFGS = [];
% optMinFunc = struct('MaxIter', maxNOfPasses);
% [XX,FVAL,exitflag] = minFunc(@LogisticLoss_for_LBFGS,w,optMinFunc,params);
% 
% f = [];
% for i = 1:1:size(THETA_LBFGS,2)
%     f(i) = objective(THETA_LBFGS(:,i));
%     fprintf('f = %.6f\n',f(i));
% end
% min_LBFGS = min(f);
% % figure(1);
% % x_ax = 1:1:maxNOfPasses*fractionPass+1;
% % x_ax_LBFGS = x_ax(1:fractionPass:maxNOfPasses*fractionPass+1); 
% % plot(x_ax_LBFGS,f(1:1:size(x_ax_LBFGS,2)),'y','LineWidth',3);
% 
% %%
% fprintf('Running basic SG method with constant step size\n');
% 
% if(data_flag == 1)
%     stepSizes = 1e-1*ones(maxIter,1);
% elseif(data_flag == 2)
%     stepSizes = 1e-3*ones(maxIter,1);
% elseif(data_flag == 3)
%     stepSizes = 1e-3*ones(maxIter,1);
% end
% w = zeros(p,1);
% THETA_OUT_SG = [];
% [THETA_OUT_SG] = SGD_logistic(w,Xt,y,lambda,stepSizes,iVals,maxNOfPasses,fractionPass);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SG,2)
%     f(i) = objective(THETA_OUT_SG(:,i));
%     fprintf('f = %.6f\n',f(i));
% end
% min_SG = min(f);
% % figure(1);
% % hold on;
% % x_ax = 1:1:maxNOfPasses*fractionPass+1;
% % plot(x_ax,f(1:1:size(x_ax,2)),'g','LineWidth',3);
% 
% %%
fprintf('Running averaged SG method with a constant step size\n');

if(data_flag == 1)
    stepSizes = 1*ones(maxIter,1);
elseif(data_flag == 2)
    stepSizes = 1e-3*ones(maxIter,1);
elseif(data_flag == 3)
    stepSizes = 1e-3*ones(maxIter,1);
end

w = zeros(p,1);
THETA_OUT_ASGD = [];
[wAvg,THETA_OUT_ASGD] = ASGD_logistic(w,Xt,y,lambda,stepSizes,iVals,maxNOfPasses,fractionPass);
% Note that ASGD_logistic is faster than SGD_logistic for sparse
% data when computing the uniform averaging

f = [];
for i = 1:1:size(THETA_OUT_ASGD,2)
    f(i) = objective(THETA_OUT_ASGD(:,i));
    fprintf('f = %.6f\n',f(i));
end
min_ASGD = min(f);
% % figure(1);
% % hold on;
% % x_ax = 1:1:maxNOfPasses*fractionPass+1;
% % plot(x_ax,f(1:1:size(x_ax,2)),'b','LineWidth',3);
% 
% %%
% fprintf('Running stochastic average gradient with constant step size\n');
% 
% if(data_flag == 1)
%     stepSize = 1;
% elseif(data_flag == 2)
%     stepSize = 1e-3;
% elseif(data_flag == 3)
%     stepSize = 1e-3;
% end
% 
% d = zeros(p,1);
% g = zeros(n,1);
% covered = int32(zeros(n,1));
% 
% w = zeros(p,1);
% THETA_OUT_SAG = [];
% [THETA_OUT_SAG] = SAG_logistic(w,Xt,y,lambda,stepSize,iVals,d,g,covered,maxNOfPasses,fractionPass);
% % {w,d,g,covered} are updated in-place
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SAG,2)
%     f(i) = objective(THETA_OUT_SAG(:,i));
%     fprintf('f = %.6f\n',f(i));
% end
% min_SAG = min(f);
% % figure(1);
% % hold on;
% % x_ax = 1:1:maxNOfPasses*fractionPass+1;
% % plot(x_ax,f(1:1:size(x_ax,2)),'r','LineWidth',3);
% 
% %%
fprintf('Running stochastic average gradient with line-search\n');

d = zeros(p,1);
g = zeros(n,1);
covered = int32(zeros(n,1));
Lmax = 1;

w = zeros(p,1);
THETA_OUT_SAG_LS = [];
[THETA_OUT_SAG_LS] = SAGlineSearch_logistic(w,Xt,y,lambda,Lmax,iVals,d,g,maxNOfPasses,fractionPass,covered);
% Lmax is also updated in-place
% You pass another argument set to int32(2) to use 2/(Lmax+n*mu) instead of
% 1/Lmax
% You can further pass in xtx = sum(X.^2,2) to avoid this computation

f = [];
for i = 1:1:size(THETA_OUT_SAG_LS,2)
    f(i) = objective(THETA_OUT_SAG_LS(:,i));
    fprintf('f = %.6f\n',f(i));
end
min_SAG_LS = min(f);
% figure(1);
% hold on;
% x_ax = 1:1:maxNOfPasses*fractionPass+1;
% plot(x_ax,f(1:1:size(x_ax,2)),'m','LineWidth',3);


%%
fprintf('Running bound method\n');

%--------------------------------------------------------------------------
lsqrIter = 10;
etaStart = 1;
eta = etaStart;
flag_Bound_Matlab = 1;
%---------------------------------------------------------------------

THETA = zeros(p,1);
THETA_BOUND = [];
THETA_BOUND(:,1) = THETA;
objInit = objective(THETA_BOUND(:,1));
fprintf('Initial f = %.6f\n',objInit);

bs = zeros(100000,1);
suma = 0;
obj = [];
maxMiniBatchPasses = fractionPass*maxNOfPasses;
objEvaluated = 0;
iter = 1;
while(objEvaluated < maxMiniBatchPasses)
    bIn = 5;
    bIncr = max(round(n), 1);
    b = min(bIn + iter*bIncr, n);
    bs(iter) = b;

    suma = suma + b;

    permtab = randperm(n,b);
    storage = length(permtab);  

    if(flag_Bound_Matlab == 1)
        ui=zeros(p,1);
        S_dod_vec = zeros(p, storage); 
        indexj = 0; 
        for j = permtab    
            logzti = log(1e-200);
            uti = zeros(p,1); 
            if(0<logzti)
                logzti = logzti + log(1.0+exp(-logzti));
            else
                logzti = log(1.0+exp(logzti));
            end
            feature = Xlabt(:,j);
            logai = THETA'*feature;
            l = feature - uti;                   
            if(logai == logzti)
                qi = 0.5*l;
            else
                qi = sqrt(tanh(0.5*(logai - logzti))/(2*(logai - logzti)))*l;
            end    
            S_dod_vec(:,indexj + 1) = qi;       
            if(logzti>logai)
                rat1 = 1.0/(1.0+exp(logai-logzti));
                rat2 = 1.0-rat1;
            else
                rat2 = 1.0/(exp(logzti-logai)+1.0);
                rat1 = 1.0-rat2;
            end
            uti = rat1*uti + rat2*feature;       
            if(logai<logzti)
                logzti = logzti + log(1.0+exp(logai-logzti));
            else
                logzti = logai + log(1.0+exp(logzti-logai));
            end
            logzti = logzti + log(1.0-exp(log(1e-200)-logzti));
            ui = ui + uti;
            ui = ui - feature;                   
            indexj = indexj+1;       
        end
    else
        [ui,S_dod_vec] = fast_semistochastic_bound(n,p,double(Xlabt),uint32(y),THETA,lambda,b,uint32(permtab),storage); 
    end
    ui = ui/n;
    lambdaEffective = lambda*(storage/n);
    ui = ui + lambdaEffective*THETA;
    
    fprod = @(x, arg)((1/n)*S_dod_vec*(S_dod_vec'*x) + lambda*x); % don't forget regularization.
    [update, ~] = pcg(fprod, -ui, [], lsqrIter );

%     S_dod_vec = S_dod_vec/sqrt(n);
%     rhs = -ui;
%     t1 = S_dod_vec'*rhs;
%     StSpI = S_dod_vec'*S_dod_vec + lambda*eye(storage);
%     t1inv = StSpI\t1;
%     u = S_dod_vec*t1inv;
%     update = (rhs - u)/lambda;

    THETA_PROP = THETA + eta*update;
    THETA = THETA_PROP;
        
    
    % Compute objective
    if (suma > n/fractionPass)
        objEvaluated = objEvaluated +1; % increase counter for objective values computed
        THETA_BOUND(:, objEvaluated+1) = THETA;
        obj = objective(THETA_BOUND(:, objEvaluated+1));
        fprintf('f = %.6f\n',obj);
        
        suma = 0;
    end

    %disp(iter);
    iter = iter + 1;
end

f = [];
for i = 1:1:size(THETA_BOUND,2)
    f(i) = objective(THETA_BOUND(:,i));
    fprintf('f = %.6f\n',f(i));
end
min_BOUND = min(f);
% figure(1);
% hold on;
% x_ax = 1:1:maxNOfPasses*fractionPass+1;
% plot(x_ax,f(1:1:size(x_ax,2)),'k','LineWidth',3);

% %% Plots
% 
% %optimum = min(min(min_SAG_LS,min(min_SAG,min(min(min_LBFGS,min_SG),min_ASGD))),min_BOUND);
% 
% optimum = min(min_SAG_LS,min(min_SAG,min(min(min_LBFGS,min_SG),min_ASGD)));
% 
% x_ax = 1:1/fractionPass:maxNOfPasses+1;
% 
% %Training
% %--------------------------------------------------------------------------
% 
% f = [];
% for i = 1:1:size(THETA_LBFGS,2)
%     f(i) = objective(THETA_LBFGS(:,i));
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(1);
% x_ax_LBFGS = x_ax(1:fractionPass:maxNOfPasses*fractionPass+1); 
% semilogy(x_ax_LBFGS,f(1:1:size(x_ax_LBFGS,2)) - optimum,'c','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SG,2)
%     f(i) = objective(THETA_OUT_SG(:,i));
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(1);
% hold on;
% semilogy(x_ax,f(1:1:size(x_ax,2)) - optimum,'g','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_ASGD,2)
%     f(i) = objective(THETA_OUT_ASGD(:,i));
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(1);
% hold on;
% semilogy(x_ax,f(1:1:size(x_ax,2)) - optimum,'b','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SAG,2)
%     f(i) = objective(THETA_OUT_SAG(:,i));
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(1);
% semilogy(x_ax,f(1:1:size(x_ax,2)) - optimum,'r','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SAG_LS,2)
%     f(i) = objective(THETA_OUT_SAG_LS(:,i));
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(1);
% hold on;
% semilogy(x_ax,f(1:1:size(x_ax,2)) - optimum,'m','LineWidth',3);
% 
% % f = [];
% % for i = 1:1:size(THETA_BOUND,2)
% %     f(i) = objective(THETA_BOUND(:,i));
% %     %fprintf('f = %.6f\n',f(i));
% % end
% % figure(1);
% % hold on;
% % semilogy(x_ax,f(1:1:size(x_ax,2)) - optimum,'m','LineWidth',3);
% 
% axis tight
% h=gca;
% set(h,'FontSize',15,'FontWeight','bold');
% title(dataset_name);
% ylabel('Objective minus optimum (training)');
% xlabel(['Effective Passes']);
% legend('LBFGS','SGD','ASGD','SAG','SAGls','SQB','Location','NorthEast');
% filename = [dataset_name,'_train'];
% saveas(gcf,filename,'fig');
% filename = [filename,'.eps'];
% print('-depsc',filename);
% 
% %Testing
% %--------------------------------------------------------------------------
% 
% f = [];
% for i = 1:1:size(THETA_LBFGS,2)
%     f(i) = objective_test(THETA_LBFGS(:,i));
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(2);
% x_ax_LBFGS = x_ax(1:fractionPass:maxNOfPasses*fractionPass+1); 
% plot(x_ax_LBFGS,f(1:1:size(x_ax_LBFGS,2)),'c','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SG,2)
%     f(i) = objective_test(THETA_OUT_SG(:,i));
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(2);
% hold on;
% plot(x_ax,f(1:1:size(x_ax,2)),'g','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_ASGD,2)
%     f(i) = objective_test(THETA_OUT_ASGD(:,i));
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(2);
% hold on;
% plot(x_ax,f(1:1:size(x_ax,2)),'b','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SAG,2)
%     f(i) = objective_test(THETA_OUT_SAG(:,i));
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(2);
% plot(x_ax,f(1:1:size(x_ax,2)),'r','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SAG_LS,2)
%     f(i) = objective_test(THETA_OUT_SAG_LS(:,i));
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(2);
% hold on;
% plot(x_ax,f(1:1:size(x_ax,2)),'m','LineWidth',3);
% 
% % f = [];
% % for i = 1:1:size(THETA_BOUND,2)
% %     f(i) = objective_test(THETA_BOUND(:,i));
% %     %fprintf('f = %.6f\n',f(i));
% % end
% % figure(2);
% % hold on;
% % plot(x_ax,f(1:1:size(x_ax,2)),'k','LineWidth',3);
% 
% axis tight
% h=gca;
% set(h,'FontSize',15,'FontWeight','bold');
% title(dataset_name);
% ylabel('Test logistic loss');
% xlabel(['Effective Passes']);
% legend('LBFGS','SGD','ASGD','SAG','SAGls','SQB','Location','NorthEast');
% filename = [dataset_name,'_test'];
% saveas(gcf,filename,'fig');
% filename = [filename,'.eps'];
% print('-depsc',filename);
% 
% %Test error
% %--------------------------------------------------------------------------
% 
% f = [];
% for i = 1:1:size(THETA_LBFGS,2)
%     f(i) = error_percent(THETA_LBFGS(:,i),X_test,y_test);
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(3);
% x_ax_LBFGS = x_ax(1:fractionPass:maxNOfPasses*fractionPass+1); 
% plot(x_ax_LBFGS,f(1:1:size(x_ax_LBFGS,2)),'c','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SG,2)
%     f(i) = error_percent(THETA_OUT_SG(:,i),X_test,y_test);
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(3);
% hold on;
% plot(x_ax,f(1:1:size(x_ax,2)),'g','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_ASGD,2)
%     f(i) = error_percent(THETA_OUT_ASGD(:,i),X_test,y_test);
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(3);
% hold on;
% plot(x_ax,f(1:1:size(x_ax,2)),'b','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SAG,2)
%     f(i) = error_percent(THETA_OUT_SAG(:,i),X_test,y_test);
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(3);
% plot(x_ax,f(1:1:size(x_ax,2)),'r','LineWidth',3);
% 
% f = [];
% for i = 1:1:size(THETA_OUT_SAG_LS,2)
%     f(i) = error_percent(THETA_OUT_SAG_LS(:,i),X_test,y_test);
%     %fprintf('f = %.6f\n',f(i));
% end
% figure(3);
% hold on;
% plot(x_ax,f(1:1:size(x_ax,2)),'m','LineWidth',3);
% 
% % f = [];
% % for i = 1:1:size(THETA_BOUND,2)
% %     f(i) = error_percent(THETA_BOUND(:,i),X_test,y_test);
% %     %fprintf('f = %.6f\n',f(i));
% % end
% % figure(3);
% % hold on;
% % plot(x_ax,f(1:1:size(x_ax,2)),'k','LineWidth',3);
% 
% axis tight
% h=gca;
% set(h,'FontSize',15,'FontWeight','bold');
% title(dataset_name);
% ylabel('Test error');
% xlabel(['Effective Passes']);
% legend('LBFGS','SGD','ASGD','SAG','SAGls','SQB','Location','NorthEast');
% filename = [dataset_name,'_error_test'];
% saveas(gcf,filename,'fig');
% filename = [filename,'.eps'];
% print('-depsc',filename);
% 
% % fprintf('The size of the dataset is %d and the dimensionality is %d\n', size_dataset, p);
% % save([filename,'_general.mat'],'size_dataset','p','maxbs','percent_of_dataset');
