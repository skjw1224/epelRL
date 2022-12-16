function [out,z4,z5,z6,pi4,pi5,pi6] = BS_new(state)

beta = 10; % 5
alpha = 5;%12

beta2 = 22; %7
alpha2 = 0.5;%15
alpha3 = 70;%50
%50 0.27 / 100  0.20


m1 = 30.8285163776493;
m2 = 0.1;
kwar = 866.880000000000;

u = zeros(2,1);
%x = [2.14040000000000;0.450000000000000;7.33999999999998;386.060000000000;14.1900000000000;-1113.50000000000];
xs = [0;0.950000000000000;380;0;0;0];
x = state - xs;
xxs = state;
xd = xdot(xxs);

k1 = 1.287e+12*exp(-9758.3/xxs(3));
k2 = 1.287e+12*exp(-9758.3/xxs(3));
k3 = 9.043e+9*exp(-8560/xxs(3));
k1h = k1*1.49362966945975;
k2h = k2*-3.91188722953745;
k3h = k3*-14.8829527778311;
k1d = 1.287e+12*exp(-9758.3/xxs(3))*(9758.3/(xxs(3)^2))*xd(3);
k2d = 1.287e+12*exp(-9758.3/xxs(3))*(9758.3/(xxs(3)^2))*xd(3);
k3d = 9.043e+9*exp(-8560/xxs(3))*(8560/(xxs(3)^2))*xd(3);
k1hd = 1.49362966945975*k1d;
k2hd = -3.91188722953745*k2d;
k3hd = -14.8829527778311*k3d;

W = x(5)*(378.05 - x(3) - xs(3)) - (k1h*x(1) + k2h*x(2) + k3h*x(1)*x(1));
Wd = u(1)*(378.05 - x(3) - xs(3)) - x(5)*xd(3) - (k1hd*x(1) +k1h*xd(1) + k2hd*x(2) + k2h*xd(2) + k3hd*x(1)*x(1) + 2*k3h*x(1)*xd(1));
x1dd = u(1)*(5.1 - x(1)) + x(5)*(-xd(1)) - k1d*x(1) - k1*xd(1) - k2d*x(1)^2 - 2*k2*x(1)*xd(1);
x2dd = -xd(3)*x(2) - x(3)*xd(2) + k1d*x(1) + k1*xd(1) - k2d*x(2) - k2*xd(2);
x3dd = Wd + m1*(xd(4) - xd(3));

k1hdd = 1.49362966945975*9758.3*(k1d*xd(3)/((x(3) + xs(3))^2) + k1*(x3dd*(x(3) + xs(3)) - 2*xd(3)^2)/((x(3) + xs(3))^3));
k2hdd = -3.91188722953745*9758.3*(k2d*xd(3)/((x(3) + xs(3))^2) + k2*(x3dd*(x(3) + xs(3)) - 2*xd(3)^2)/((x(3) + xs(3))^3));
k3hdd = -14.8829527778311*8560*(k3d*xd(3)/((x(3) + xs(3))^2) + k3*(x3dd*(x(3) + xs(3)) - 2*xd(3)^2)/((x(3) + xs(3))^3));

pi5 = (k1*x(1) - k2*x(2) - k2*xs(2) + beta*x(2))/(x(2) + xs(2));
pi5d = ((k1d*x(1) + k1*xd(1) - k2d*x(2) - k2*xd(2) - k2d*xs(2) + beta*xd(2))*(x(2) + xs(2)) - xd(2)*(k1*x(1) - k2*x(2) - k2*xs(2) + beta*x(2)))/((x(2) + xs(2))^2);
z5 = x(5) - pi5;
u(1) = pi5d - alpha*z5 + x(2)*(x(2) + xs(2));

Wdd = -2*u(1)*xd(3) - x(5)*x3dd - (k1hdd*x(1) + 2*k1hd*xd(1) + x1dd*k1h + k2hdd*x(2) + 2*k2hd*xd(2) + x2dd*k2h + k3hdd*x(1)^2 + 4*k3hd*x(1)*xd(1) + 2*k3h*(xd(1)*xd(1) + x(1)*x1dd)); 
pi4 = x(3) + xs(3) + (-W - beta2*x(3))/m1;
z4 = x(4) - pi4;
pi4d = xd(3) + (-Wd - beta2*xd(3))/m1;

pi4dd = x3dd + (-Wdd - beta2*x3dd)/m1;
pi6 = -kwar*(x(3) + xs(3) - x(4)) + pi4d/m2 + (-alpha2*z4 - m1*x(3))/m2;

z6 = x(6) - pi6;
pi6d = -kwar*(xd(3) - xd(4)) + pi4dd/m2 + (-alpha2*(xd(4) - pi4d) - m1*xd(3))/m2;
u(2) = pi6d - alpha3*z6 - m2*z4;

out = u;

function out = xdot(x)
    CA = x(1);
    CB = x(2);
    T = x(3);
    TK = x(4);
    VdotVR = x(5);
    QKdot = x(6);

    k10 = 1.287e+12; k20 = 1.287e+12; k30 = 9.043e+9;
    E1 = -9758.3; E2 = -9758.3; E3 = -8560;
    CA0 = 5.1; T0 = 378.05;
    rho = 0.9342; Cp = 3.01; kw = 4032; AR = 0.215; VR = 10; mk = 5; CpK = 2 ;
    delHRab = 4.2; delHRbc = -11.0; delHRad = -41.85 ;

    k11 = k10 * exp(E1 / T);
    k22 = k20 * exp(E2 / T);
    k33 = k30 * exp(E3 / T);

    out = [VdotVR * (CA0 - CA) - k11 * CA - k33 * CA^2;
        -VdotVR * CB + k11 * CA - k22 * CB;
        VdotVR * (T0 - T) - (k11 * CA * delHRab + k22 * CB * delHRbc + k33 * CA^2. * delHRad) / (rho * Cp) + (kw * AR) / (rho * Cp * VR) * (TK - T);
        (QKdot + (kw * AR) * (T - TK)) / (mk * CpK)];
end

end