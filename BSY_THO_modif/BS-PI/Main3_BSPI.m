%%
clear all
close all
clc

N_plant = 10;    % plant iteration
n = 180;        % time horizon
delt = 20/3600;
w = randn(2, n, N_plant);       % disturbance perturbation

sum_cost = zeros(2, N_plant);
time = zeros(1, N_plant);

% Hyper-parameters setting
Q = 1 * diag([0, 0, 10, 5/100, 0, 0, 0]);
Rbs = 1e-11*[1000000 0 ; 0 1];
R = zeros(2, 2, n);
lambda = 1;

B_2 = 0.3;

x0 = [0.0, 2.1404, 1.4, 387.34, 386.06, 14.19, -1113.5]' ;
xs = [0; 0; 0.95; 380; 0; 0; 0];
us = [0.2;-880];

% BS
dx_BS = zeros(7, n+1, N_plant); 
u_BS = zeros(2, n, N_plant);

% PI + BS
dx = zeros(7, n+1, N_plant); 
u = zeros(2, n);
delu = zeros(2, n, N_plant);
piu = zeros(2, n, N_plant);
pig = zeros(2, 2,n);
u_BSPI = zeros(2, n, N_plant);


% Plant iteration 
for p = 1 : N_plant 
    
    
    %% Backstepping
    
    % Initialization 
    dx_BS(:, 1, p) = x0;
    x_BS = x0;
    delu_BS = zeros(2, n);
    piu_BS = zeros(2, n);
    pig_BS = zeros(2, 2, n);
    r_BS = zeros(1, n);
    rx_BS = zeros(1, n);
    ru_BS = zeros(1, n);
    rbs = zeros(1, n);
    
    for i = 1 : n
        
        i

        % Back-stepping 
        [u_BS(:, i, p), z4(i), z5(i), z6(i), pi4(i), pi5(i), pi6(i)] = BS_new(x_BS(2:end));

        % Path integral
        piu_BS(:, i) = zeros(2,1);
        pig_BS(1, 1, i) = -z5(i); 
        pig_BS(2, 2, i) = -z6(i);

        % Disturbance model
        B(:,:,i) = [3*abs(dx_BS(6, i, p))/200 + B_2, 0 ; 0, 3*abs(dx_BS(7, i, p))/80 + 3000];          
        R(:,:,i) = lambda*[z5(i)^2, 0 ; 0 z6(i)^2]*(B(:,:,i)*B(:,:,i))^(-1);     

        % Cost
        rx_BS(i) = (x_BS - xs)' * Q * (x_BS - xs); 
        ru_BS(i) = piu_BS(:,i)'* R(:,:,i) * piu_BS(:,i);  
        rbs(i) = (u_BS(:, i, p) - us)' * Rbs * (u_BS(:, i, p) - us);
        r_BS(i) =  delt * (rx_BS(i) + ru_BS(i) + rbs(i)); 

        % Deriving input
        u_BS(:, i, p); disp(u_BS(:, i, p))                                         % BS
        delu_BS(:, i) = pig_BS(:, :, i) * piu_BS(:, i); disp(delu_BS(:,i))    % PI
        U_BS = u_BS(:, i, p) + delu_BS(:, i);                                    % BS + PI

        % Solving ODE
        dx_BS(:, i+1, p) = plant(x_BS, U_BS)' + [0;0;0;0;0;B(:, :, i) * sqrt(delt) * w(:, i, p)];
        x_BS = dx_BS(:, i+1, p);
        
    end

    sum_cost(1, p) = delt*sum(rx_BS + rbs);

    
    
    %% BS + PI 
    
%     Initialization
    dx(:, 1, p) = x0;
    x = x0;
    r = zeros(1, n);
    rx = zeros(1, n);
    ru = zeros(1, n);
    rbs = zeros(1, n);
    
    tic  % 시간 측정 시작
    
    for i = 1 : n
        
        i
        
        % Back-stepping 
        [u(:, i), z4(i), z5(i), z6(i), pi4(i), pi5(i), pi6(i)] = BS_new(x(2:end));

        % Path integral
        piu(:, i, p) = PI(dx(:, i, p), n-i+1); piu(:, i, p)          % PI (MPC fashion)
%         piu(:, i, p) = PI(dx(:,i),n-i+1); piu(:,i)       % PI (본래 방법) 
        pig(1,1,i) = -z5(i); 
        pig(2,2,i) = -z6(i);

        % Disturbance model
        B(:,:,i) = [3*abs(dx(6, i, p))/200 + B_2, 0 ; 0, 3*abs(dx(7, i, p))/80 + 3000];          
        R(:,:,i) = lambda * [z5(i)^2, 0 ; 0 z6(i)^2] * (B(:,:,i)*B(:,:,i))^(-1);   

        % Cost
        rx(i) = (x - xs)'* Q * (x - xs); 
        ru(i) = piu(:, i, p)' * R(:,:,i) * piu(:, i, p);  
        rbs(i) = (u(:, i) - us)' * Rbs * (u(:, i) - us);
        r(i) =  delt * (rx(i) + ru(i) + rbs(i)); 

        % Deriving input
        u(:, i); disp(u(:, i))                                % BS
        delu(:, i, p) = pig(:, :, i) * piu(:, i, p); disp(delu(:, i, p))    % PI
        u_BSPI(:, i, p) = u(:, i) + delu(:, i, p);                   % BS + PI
        u2 = max(u_BSPI(2, i, p), -2.0 * 10^5) ;
        u_BSPI(2, i, p) = u2 ;
        
        % Solving ODE
        dx(:, i+1, p) = plant(x, u_BSPI(:, i, p))' + [0; 0; 0; 0; 0; B(:,:,i) * sqrt(delt) * w(:, i, p)];
        x = dx(:, i+1, p);
    
    end
    
    time(p) = toc;
    sum_cost(2, p) = delt*sum(rx + rbs);
    
    % Plotting (PI-BS)
    t = linspace(0, 1, 181);
    t_u = linspace(0, 1-1/180, 180);  
    
    figure
    subplot(2,2,1)
    plot(t_u, u_BSPI(1, :, p))
    hold on 
    plot(t_u, u_BS(1, :, p))
    title('$\Delta \frac{\dot{V}}{VR}$', 'interpreter', 'latex'); 
    xlabel('Time(hr)');  grid on
    subplot(2,2,2)
    plot(t_u, u_BSPI(2, :, p))
    hold on 
    plot(t_u, u_BS(2, :, p))
    title('$\Delta \dot{Q}_K$', 'interpreter', 'latex'); xlabel('Time(hr)'); 
    legend('BS+PI', 'BS'); grid on
    drawnow
    
    figure
    subplot(2,3,1)
    plot(t, dx(2, :, p)) 
    hold on
    plot(t, dx_BS(2, :, p))
    title("C_A"); xlabel('Time(hr)'); grid on
    subplot(2,3,2)
    plot(t, dx(3, :, p))
    hold on
    plot(t, dx_BS(3, :, p))
    title('C_B'); xlabel('Time(hr)'); grid on
    subplot(2,3,3)
    plot(t, dx(4, :, p))
    hold on
    plot(t, dx_BS(4, :, p))
    title('T'); xlabel('Time(hr)'); legend('BS+PI', 'BS'); grid on
    subplot(2,3,4)
    plot(t, dx(5, :, p))
    hold on
    plot(t, dx_BS(5, :, p))
    title('T_K'); xlabel('Time(hr)'); grid on
    subplot(2,3,5)
    plot(t, dx(6, :, p))
     hold on
    plot(t, dx_BS(6, :, p))
    title('$\frac{\dot{V}}{VR}$', 'interpreter', 'latex')
    xlabel('Time(hr)'); grid on
    subplot(2,3,6)
    plot(t, dx(7, :, p))
     hold on
    plot(t, dx_BS(7, :, p))
    title('$\dot{Q}_K$', 'interpreter', 'latex'); 
    xlabel('Time(hr)'); grid on
    drawnow

% Plotting (BS)

% figure
% subplot(2,2,1)
% plot(u_BS(1, :, p))
% title("{\Delta} VdotVR"); xlabel('Time(hr)')
% subplot(2,2,2)
% plot(u_BS(2, :, p))
% title("{\Delta} QKdot"); xlabel('Time(hr)')
% drawnow
% 
% figure
% subplot(2,3,1)
% plot(dx_BS(2, :, p))
% title("C_A"); xlabel('Time(hr)')
% subplot(2,3,2)
% plot(dx_BS(3, :, p))
% title('C_B'); xlabel('Time(hr)')
% subplot(2,3,3)
% plot(dx_BS(4, :, p))
% title('T'); xlabel('Time(hr)')
% subplot(2,3,4)
% plot(dx_BS(5, :, p))
% title('TK'); xlabel('Time(hr)')
% subplot(2,3,5)
% plot(dx_BS(6, :, p))
% title('VdotVR'); xlabel('Time(hr)')
% subplot(2,3,6)
% plot(dx_BS(7, :, p))
% title('QKdot'); xlabel('Time(hr)')
% drawnow

end

save('dx_BS')
save('u_BS')
save('dx')
save('u_BSPI')
save('sum_cost')
save('time')


