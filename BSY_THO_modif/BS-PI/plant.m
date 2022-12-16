function dx = plant(x, u)

    dt = 20/3600 ;
    tspan = [0, dt]';  
    [t, dx] = ode15s(@(t, x) VDV_reactor(t, x, u'), tspan, x) ;
    dx = dx(end,:);
    
    function dx = VDV_reactor(t, x, u)

        CA = x(2); 
        CB = x(3); T = x(4); TK = x(5); VdotVR = x(6); QKdot = x(7);
        dVdotVR = u(1); dQKdot = u(2);

        k10 = 1.287e+12; k20 = 1.287e+12; k30 = 9.043e+9;
        E1 = -9758.3; E2 = -9758.3; E3 = -8560;
        CA0 = 5.1; T0 = 378.05;
        rho = 0.9342; Cp = 3.01; kw = 4032; AR = 0.215; VR = 10; mk = 5; CpK = 2 ;
        delHRab = 4.2; delHRbc = -11.0; delHRad = -41.85 ;

        k1 = k10 * exp(E1 / T);
        k2 = k20 * exp(E2 / T);
        k3 = k30 * exp(E3 / T);

        dx = [1;
              VdotVR * (CA0 - CA) - k1 * CA - k3 * CA^2;
              -VdotVR * CB + k1 * CA - k2 * CB;
              VdotVR * (T0 - T) - (k1 * CA * delHRab + k2 * CB * delHRbc + k3 * CA^2. * delHRad) / (rho * Cp) + (kw * AR) / (rho * Cp * VR) * (TK - T);
              (QKdot + (kw * AR) * (T - TK)) / (mk * CpK);
              dVdotVR;
              dQKdot]; 
    end
end