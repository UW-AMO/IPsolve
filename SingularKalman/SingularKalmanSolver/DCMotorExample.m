%DC MOTOR

%%Impulsive inputs
G = [0.7047 0;0.08437 1];
H = [0 1];
Q = .01*[11.81;0.625]*[11.81 0.625];
R = .1^2;
n = 2;
m = 1;

A=[0.704 0;0.08437 1];
Vo=0*eye(2);
xo=[0 0]';
N=200;
mu=0.01;%probability of spike

    Mu=zeros(1,N);
    while sum(Mu>0)==0
        for i=1:N%generation of x_1,x_2,... and y_1,y_2,...
            g(:,:,i)=A;
            Mu(i)=rand<mu;
            qt(:,:,i)=Mu(i)*Q*100;%qtrue
            q(:,:,i)=mu*Q*100;
            h(:,:,i)=[0 1];
            rv(:,:,i)=.01;
            if i==1
                Xt(:,i)=g(:,:,i)*xo+mvnrnd([0 0],qt(:,:,i),1)';
                temp = Xt(:,1) - g(:,:,1)*xo;
                dt_true(1) = -1*temp(1)/11.81;
            else
                Xt(:,i)=g(:,:,i)*Xt(:,i-1)+mvnrnd([0 0],qt(:,:,i),1)';
                temp = Xt(:,i) - g(:,:,i)*Xt(:,i-1);
                dt_true(i) = -1*temp(1)/11.81;
            end
            y(i)=h(:,:,i)*Xt(:,i)+sqrt(.01)*randn;
            dati(i,:)=[i y(i)];
        end
    end
    
    
    
 
    
 [Lagsoln, Relaxsoln, Robustsoln] = SingularKalmanRunner(G, H, Q, R, [0;0], N, dati(:,2), 100, 'w');
 d_tLag = Lagsoln(1,:)/11.81;
 d_tRelax = Relaxsoln(1,:)/11.81;
 d_tRobust = Robustsoln(1,:)/11.81;
% [D, A, hatw] = SingularKalmanSetup(G, H, Q, R, [0;0], N, dati(:,2));
% Lagwsoln = LagrangianSolver(A, D, hatw, n,m, N, 'w');
% d_tLag = Lagwsoln(1,:)/11.81;
 t = 1:length(d_tLag);


plot(t',dt_true, 'k', t', d_tLag, 'r--o', t', d_tRelax, 'b--*', t', d_tRobust, 'm--+' )
legend('True error', 'Lagrangian error', 'Relaxed error', 'Robust error')



