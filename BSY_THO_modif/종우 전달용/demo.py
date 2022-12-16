import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class InitialControl(object):
    def __init__(self):
        self.E1 = -9758.3
        self.E2 = -9758.3
        self.E3 = -8560.

        self.rho = 0.9342  # (KG / L)
        self.Cp = 3.01  # (KJ / KG K)
        self.kw = 4032.  # (KJ / h M ^ 2 K)
        self.AR = 0.215  # (M ^ 2)
        self.VR = 10.  # L
        self.mk = 5.  # (KG)
        self.CpK = 2.0  # (KJ / KG K)

        self.CA0 = 5.1  # mol / L
        self.T0 = 378.05  # K

        self.k10 = 1.287e+12
        self.k20 = 1.287e+12
        self.k30 = 9.043e+9

        self.delHRab = 4.2  # (KJ / MOL)
        self.delHRbc = -11.0  # (KJ / MOL)
        self.delHRad = -41.85  # (KJ / MOL)

        self.dt = 20 / 3600
        self.tT = 3600 / 3600
        self.dx_eval = self.system_functions

    def xdot(self, state):

        k10, k20, k30, E1, E2, E3 = self.k10, self.k20, self.k30, self.E1, self.E2, self.E3
        delHRab, delHRbc, delHRad = self.delHRab, self.delHRbc, self.delHRad
        CA0, T0 = self.CA0, self.T0
        rho, Cp, kw, AR, VR = self.rho, self.Cp, self.kw, self.AR, self.VR
        mk, CpK = self.mk, self.CpK

        x = state
        CA, CB, T, TK, VdotVR, QKdot = x[0]

        k1 = k10 * np.exp(E1 / T)
        k2 = k20 * np.exp(E2 / T)
        k3 = k30 * np.exp(E3 / T)

        xdot = [VdotVR * (CA0 - CA) - k1 * CA - k3 * CA ** 2.,
               -VdotVR * CB + k1 * CA - k2 * CB,
               VdotVR * (T0 - T) - (k1 * CA * delHRab + k2 * CB * delHRbc + k3 * CA ** 2. * delHRad) / (rho * Cp)
               + (kw * AR) / (rho * Cp * VR) * (TK - T),
               (QKdot + (kw * AR) * (T - TK)) / (mk * CpK)]

        xdot = np.array([xdot])

        return xdot

    def backstepping(self, state):

        beta = 10
        alpha = 5
        beta2 = 22
        alpha2 = 0.5
        alpha3 = 70

        m1 = 30.8285163776493
        m2 = 0.1
        kwar = 866.880000000000

        u = np.zeros((2, 1))
        xs = np.array([[0, 0.95, 380, 0, 0, 0]])
        x = state - xs
        xxs = state
        xd = self.xdot(xxs)

        x = x.T
        xxs = xxs.T
        xd = xd.T
        xs = xs.T

        k10, k20, k30, E1, E2, E3 = self.k10, self.k20, self.k30, self.E1, self.E2, self.E3

        k1 = k10 * np.exp(E1 / xxs[2])
        k2 = k20 * np.exp(E2 / xxs[2])
        k3 = k30 * np.exp(E3 / xxs[2])

        k1h = k1 * 1.49362966945975
        k2h = k2 * -3.91188722953745
        k3h = k3 * -14.8829527778311
        k1d = 1.287e+12 * np.exp(-9758.3 / xxs[2]) * (9758.3 / (xxs[2] ** 2)) * xd[2]
        k2d = 1.287e+12 * np.exp(-9758.3 / xxs[2]) * (9758.3 / (xxs[2] ** 2)) * xd[2]
        k3d = 9.043e+9 * np.exp(-8560 / xxs[2]) * (8560 / (xxs[2] ** 2)) * xd[2]
        k1hd = 1.49362966945975 * k1d
        k2hd = -3.91188722953745 * k2d
        k3hd = -14.8829527778311 * k3d

        W = x[4] * (378.05 - x[2] - xs[2]) - (k1h * x[0] + k2h * x[1] + k3h * x[0] * x[0])
        Wd = u[0] * (378.05 - x[2] - xs[2]) - x[4] * xd[2] - (k1hd * x[0] + k1h * xd[0] + k2hd * x[1] + k2h * xd[1]
              + k3hd * x[0] * x[0] + 2 * k3h * x[0] * xd[0])

        x1dd = u[0] * (5.1 - x[0]) + x[4] * (-xd[0]) - k1d * x[0] - k1 * xd[0] - k2d * x[0] ** 2 - 2 * k2 * x[0] * xd[0]
        x2dd = -xd[2] * x[1] - x[2] * xd[1] + k1d * x[0] + k1 * xd[0] - k2d * x[1] - k2 * xd[1]
        x3dd = Wd + m1 * (xd[3] - xd[2])

        k1hdd = 1.49362966945975 * 9758.3 * (k1d * xd[2] / ((x[2] + xs[2]) ** 2)
                + k1 * (x3dd * (x[2] + xs[2]) - 2 * xd[2] ** 2) / ((x[2] + xs[2]) ** 3))
        k2hdd = -3.91188722953745 * 9758.3 * (k2d * xd[2] / ((x[2] + xs[2]) ** 2)
                  + k2 * (x3dd * (x[2] + xs[2]) - 2 * xd[2] ** 2) / ((x[2] + xs[2]) ** 3))
        k3hdd = -14.8829527778311 * 8560 * (k3d * xd[2] / ((x[2] + xs[2]) ** 2)
                 + k3 * (x3dd * (x[2] + xs[2]) - 2 * xd[2] ** 2) / ((x[2] + xs[2]) ** 3))

        pi5 = (k1 * x[0] - k2 * x[1] - k2 * xs[1] + beta * x[1]) / (x[1] + xs[1])
        pi5d = ((k1d * x[0] + k1 * xd[0] - k2d * x[1] - k2 * xd[1] - k2d * xs[1] + beta * xd[1]) * (x[1] + xs[1])
                - xd[1] * (k1 * x[0] - k2 * x[1] - k2 * xs[1] + beta * x[1])) / ((x[1] + xs[1]) ** 2)
        z5 = x[4] - pi5
        u[0] = pi5d - alpha * z5 + x[1] * (x[1] + xs[1])

        Wdd = -2 * u[0] * xd[2] - x[4] * x3dd - (k1hdd * x[0] + 2 * k1hd * xd[0] + x1dd * k1h + k2hdd * x[1]
             + 2 * k2hd * xd[1] + x2dd * k2h + k3hdd * x[0] ** 2 + 4 * k3hd * x[0] * xd[0]
             + 2 * k3h * (xd[0] * xd[0] + x[0] * x1dd))
        pi4 = x[2] + xs[2] + (-W - beta2 * x[2]) / m1
        z4 = x[3] - pi4
        pi4d = xd[2] + (-Wd - beta2 * xd[2]) / m1

        pi4dd = x3dd + (-Wdd - beta2 * x3dd) / m1
        pi6 = -kwar * (x[2] + xs[2] - x[3]) + pi4d / m2 + (-alpha2 * z4 - m1 * x[2]) / m2

        z6 = x[5] - pi6
        pi6d = -kwar * (xd[2] - xd[3]) + pi4dd / m2 + (-alpha2 * (xd[3] - pi4d) - m1 * xd[2]) / m2
        u[1] = pi6d - alpha3 * z6 - m2 * z4

        return u, z4, z5, z6, pi4, pi5, pi6

    def system_functions(self, *args):

        t, x, u = args

        k10, k20, k30, E1, E2, E3 = self.k10, self.k20, self.k30, self.E1, self.E2, self.E3
        delHRab, delHRbc, delHRad = self.delHRab, self.delHRbc, self.delHRad
        CA0, T0 = self.CA0, self.T0
        rho, Cp, kw, AR, VR = self.rho, self.Cp, self.kw, self.AR, self.VR
        mk, CpK = self.mk, self.CpK

        CA, CB, T, TK, VdotVR, QKdot = np.reshape(x, [-1, ])
        dVdotVR, dQKdot = np.reshape(u, [-1, ])

        k1 = k10 * np.exp(E1 / T)
        k2 = k20 * np.exp(E2 / T)
        k3 = k30 * np.exp(E3 / T)

        dx = [VdotVR * (CA0 - CA) - k1 * CA - k3 * CA ** 2.,
              -VdotVR * CB + k1 * CA - k2 * CB,
              VdotVR * (T0 - T) - (k1 * CA * delHRab + k2 * CB * delHRbc + k3 * CA ** 2. * delHRad) /
              (rho * Cp) + (kw * AR) / (rho * Cp * VR) * (TK - T),
              (QKdot + (kw * AR) * (T - TK)) / (mk * CpK),
              dVdotVR,
              dQKdot]  # shape [6, n_batch]

        return dx

    def step(self, time, state, action):

        t = time
        x = state
        u = action

        # Integrate ODE
        dx = lambda t, x: self.dx_eval(t, x, u)
        xvec = np.reshape(x, [-1, ])
        sol_x = solve_ivp(dx, [t, t + self.dt], xvec, method='LSODA')
        xplus = np.reshape(sol_x.y[:, -1], [1, -1])

        return xplus


BS = InitialControl()
x0 = np.array([[2.1404, 1.40, 387.34, 386.06, 14.19, -1113.5]])
t0 = 0
dt = 20 / 3600
tT = 3600 / 3600
N = int(tT / dt)

x = x0
t = t0

x_trajectory = np.zeros((N+1, len(x0[0])))
x_trajectory[0] = x0

for i in range(int(N)):
    u, _, _, _, _, _, _ = BS.backstepping(x)
    dx = BS.step(t, x, u)
    x_trajectory[i+1] = dx
    t += dt
    x = dx

time = range(N+1)
fig, axs = plt.subplots(3, 1)
label = [r'$C_{A}$', r'$C_{B}$', r'$T$', r'$T_{Q}$', r'$\frac{v}{V_{R}}$', r'$Q$']
for j in range(len(label)):
    axs = fig.add_subplot(3, 2, j+1)
    axs.save(time, x_trajectory[:, j])
plt.show()
print("hi")
