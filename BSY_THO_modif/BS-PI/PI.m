function out = PI(x_in, N)

    Epi = 3;
    lambda = 1/100;
    ex = [100; 1000];   % Exploration

    Q = 1*diag([0,0,10,5/100,0,0,0]); %%%%%%%%%%%%%
    Rbs = 1e-11*[1000000 0 ; 0 1]; %%%%%%%%%
    
    xm = [2.80, 0.93, 378, 372, 17.0, -6500]' ;
    xM = [3.20, 0.97, 382, 378, 23.0, -3500]' ;
    
    B_2 = 0.3; 
    
    Up = zeros(2, Epi);
    Below = zeros(Epi, 1);
    
    % Episode iteration
    for p = 1 : Epi
        delt = 20/3600;
        dx = zeros(7,N+1); 
        u = zeros(2,N);
        piu = zeros(2,N);
        pig = zeros(2,2,N);
        w = zeros(2,N);
        
        % cost
        r = zeros(1,N);
        rx = zeros(1, N);
        ru = zeros(1, N);
        rbs = zeros(1, N);
        
        R = zeros(2,2,N);
        dx(:,1) = x_in;
        xs = [0; 0; 0.95; 380; 0; 0; 0];
        
        [dum_u,dum_z4,dum_z5,dum_z6,dum_pi4,dum_pi5,dum_pi6] = BS_new(x_in(2:end));
        Gc = [-dum_z5 0; 0 -dum_z6];
        
            for i = 1 : N
                
                if i > 5
                    if all(dx(2:end,i) < xM)
                        if all(dx(2:end,i) > xm)
                            %disp(i)
                            rx(i:end) = ter_cost(dx(:,i))*ones(1,N-i+1);
                            rbs(i:end) = zeros(1,N-i+1);
                            break
                        end
                    end
                end
                
                % Back-stepping 
                [u(:,i), z4(i), z5(i), z6(i), pi4(i), pi5(i), pi6(i)] = BS_new(dx(2:end,i));
                
                % Path integral
                if i == 1
                    piu(:,i) = [ex(1)*randn(1); ex(2)*randn(1)];    % 한 스텝만 PI input 사용
                else
                    piu(:,i) = zeros(2,1);
                end
                pig(1,1,i) = -z5(i); pig(2,2,i) = -z6(i);
                
                % Disturbance model
                B = [3*abs(dx(6,i))/200 + B_2, 0 ; 0, 3*abs(dx(7,i))/80 + 3000]; 
                R(:,:,i) = lambda*[z5(i)^2, 0 ; 0 z6(i)^2]*(B*B)^(-1);
                w(:,i) = randn(2,1);     
                
                % Cost
                rx(i) = (dx(:,i) - xs)' * Q * (dx(:,i) - xs); 
                ru(i) = piu(:,i)'* R(:,:,i) * piu(:,i);  
                rbs(i) = u(:,i)' * Rbs * u(:,i);
                r(i) =  delt * (rx(i) + ru(i)); 
                
                % Solving ODE
                U = u(:,i) + pig(:,:,i)*piu(:,i);
                dx(:, i+1) = plant(dx(:,i), U)' + [0;0;0;0;0;B*sqrt(delt)*w(:,i)];
                %sum(abs(dx(:,i+1)))
                if sum(abs(dx(:,i+1))) > 10^7
                    disp('breaking')
                    rx = 100*ones(1,N);
                    break
                end
            end
            
        Up(:,p) = exp(-sum((rx+rbs)*delt)/lambda)*([3*abs(dx(6,1))/200 + B_2, 0 ; 0, 3*abs(dx(7,1))/80 + 3000]*w(:,1)/sqrt(delt) + pig(:,:,1)*piu(:,1)); %%%%%%%%%%%%%%%%%
        Below(p) = exp(-sum((rx+rbs)*delt)/lambda);
        if isnan(Up(1,p)) | isnan(Up(2,p))
            Up(:,p) = zeros(2,1);
            Below(p) = 0;
        end
        
    end
    %Up
    %Below
    u_pi = Gc^(-1)*[sum(Up')/sum(Below)]';
%     u_pi = Gc^(-1)*[Up'/sum(Below)]';   % Case : Epi = 1
    out = u_pi;
