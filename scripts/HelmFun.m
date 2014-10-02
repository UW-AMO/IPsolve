%case 2  % Helmholtz multiple sources, pxm restrictions
function [Fret Jret] = HelmFun(x, params)
%           addpath ..\Helmholtz



Q = params.Q;
q = params.q;

F_0 = params.F_0;
rho = params.rho;
omega = params.omega;

%------------------------------------------------------------
% compute forward and get data
F = vec(Q' * solveHelmholtz(Helmholtz3DOperator(x, rho, omega, params), q, params));

F = (F - F_0); % Sasha: removed transpose

%Fret = [real(F); imag(F)];
Fret = real(F);

%F = func_only_F(x, params);
%n = length(x);
%m = length(F);
%         I = eye(n);
%------------------------------------------------------------
% compute Jacobian
%J = Jacobian_only_J(x, params);

%
%         if ~isempty(old_J)
%             for i = 1:m
%                 y = (J(i,:)-old_J(i,:))';
%                 delta = x-old_x;
%                 if SR1
%                     if abs(((y - squeeze(H(i,:,:)) * delta)' * delta)) > sqrt(eps)
%                         H(i,:,:) = squeeze(H(i,:,:)) + ( 1/((y - squeeze(H(i,:,:)) * delta)' * delta)) ...
%                             *(y - squeeze(H(i,:,:)) * delta)*(y - squeeze(H(i,:,:)) * delta)';
%                     end
%                 else
%                     H(i,:,:) = squeeze(H(i,:,:)) * (y*y')/ y' * delta  - (squeeze(H(i,:,:)) * delta) * (squeeze(H(i,:,:)) * delta)'/(delta' * squeeze(H(i,:,:)) * delta);
%                 end
%             end
%         end


rho = params.rho;
omega = params.omega;
[~,J] = computeHelmholtzSensitivities(Helmholtz3DOperator(x, rho, omega, params), omega, params);

%Jret = [real(J); imag(J)];
Jret = real(J);
