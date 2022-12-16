import numpy as np
import random
import utils
import torch

#from replay_buffer import ReplayBuffer
# from ou_noise import OU_Noise

# System
# dx = (f(x) + g(x)u)dt + B(x)w dBt
# u = u_ini + u_pi


class PI(object):
    def __init__(self, env, initial_controller, device):
        self.Env = env
        self.initial_controller = initial_controller
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.xmin = env.xmin
        self.xmax = env.xmax
        self.umin = env.umin
        self.umax = env.umax
        self.parameter_lambda = 0.01
        self.device = device
        self.Q = np.diag([0., 0., 10., 0.05, 0., 0., 0.])
        self.R_bs = (10**-11) * np.diag([1000000, 1.])
        self.Target_state = np.array([0., 0., 0.95, 380., 0., 0., 0.])

    def Initial_ctrl(self, x):
        u, z4, z5, z6, pi4, pi5, pi6 = self.initial_controller.backstepping(x)
        return u, z5, z6

    def ctrl(self, epi, step, state, input):
        action = self.choose_action(epi, state, self.Env.nT - step)
        return action

    def choose_action(self, episode, state, horizon):

        x_in_torch = utils.descale(state, self.xmin, self.xmax)  # torch
        x_in = x_in_torch.detach().numpy()   # torch -> numpy, descaled
        N = horizon

        """PI 하이퍼 파라미터"""
        epi_number = 100
        exploration_rate = np.array([[30, 300]])  # Heuristic 한 부분 (0으로 줘도 됨 = 빼도 됨)

        Up = np.zeros((epi_number, self.a_dim))     # epi_number x 2
        Below = np.zeros((epi_number, 1))           # epi_number x 1
        u_pi = np.zeros(self.a_dim)

        for k in range(epi_number):
            print("MC_episode: ", k)
            state_cost = np.zeros(N)
            initial_control_cost = np.zeros(N)
            total_state_cost = np.zeros(N)
            piu = np.zeros((N, self.a_dim))
            pig = np.zeros((N, self.a_dim, self.a_dim))
            w = np.zeros((N, self.a_dim))
            u_BS_sum, dum_z5, dum_z6 = self.Initial_ctrl(x_in)        # backsetpping control
            dum_z5 = np.squeeze(dum_z5)
            dum_z6 = np.squeeze(dum_z6)
            Gc = np.array([[-dum_z5, 0], [0, -dum_z6]])
            x = x_in

            for j in range(N):

                # Initial control
                u_BS, z5, z6 = self.Initial_ctrl(x)
                z5 = np.squeeze(z5)
                z6 = np.squeeze(z6)
                pig[j, :, :] = np.array([[-z5, 0], [0, -z6]])

                if j == 0:
                    piu[j] = exploration_rate * np.random.normal(0, 1, (1, self.a_dim))  # 1 x 2
                else:
                    piu[j] = np.zeros((1, self.a_dim))  # 1 x 2

                # Cost
                state_cost[j] = (x - self.Target_state) @ self.Q @ (x - self.Target_state).T  # rx(i)
                initial_control_cost[j] = u_BS.T @ self.R_bs @ u_BS  # rbs(i)
                total_state_cost[j] = self.Env.dt * (state_cost[j] + initial_control_cost[j])  # r(i)

                # Disturbance model
                u_MC = u_BS + pig[j, :, :] @ np.array([piu[j]]).T    # 2 x 1
                w[j] = np.random.normal(0, 1, (1, self.a_dim))  # 1 x 2
                Browian_motion = self.B(x)    # 7 x 2

                # Running Plant (Monte-Carlo search)
                u_MC = u_MC.T
                u_MC = u_MC.astype(np.float32)
                u_MC_torch = torch.from_numpy(u_MC)     # descaled
                u_MC_torch = utils.scale(u_MC_torch, self.umin, self.umax)  # torch, scaled
                x_torch = torch.from_numpy(x)
                x_torch = utils.scale(x_torch, self.xmin, self.xmax)    # torch, scaled

                xplus_torch, _, u_MC_torch, _, _, _ = self.Env.step(x_torch, u_MC_torch)

                xplus_torch = utils.descale(xplus_torch, self.xmin, self.xmax)
                xplus = xplus_torch.detach().numpy()    # 1 x 7
                xplus += w[j] @ Browian_motion.T * np.sqrt(self.Env.dt) # (1 x 2) @ (2 x 7) * scalar

                x = xplus

            Up[k, :] = np.exp(-np.sum(total_state_cost) / self.parameter_lambda) * (w[0] @ Browian_motion[5: ].T / np.sqrt(self.Env.dt) + piu[0] @ pig[0, :, :].T)
            Below[k] = np.exp(-np.sum(total_state_cost) / self.parameter_lambda)

        # Input derived by PI
        u_pi = sum(Up) / sum(Below) @ np.linalg.inv(Gc)   # 1 x 2

        # Integrating u_BS with u_PI
        u_BSPI = u_BS_sum.T + u_pi @ Gc.T
        u_BSPI = u_BSPI.astype(np.float32)
        u_BSPI_torch = torch.from_numpy(u_BSPI)
        u_BSPI_torch = utils.scale(u_BSPI_torch, self.umin, self.umax)

        return u_BSPI_torch   # 1 x 2

    def B(self, state):
        Browian_motion_coefficient = np.zeros((self.Env.s_dim, self.Env.a_dim)) # 7 x 2
        Browian_motion_coefficient[5, 0] = 3 * state[0][5]/200 + 0.3
        Browian_motion_coefficient[6, 1] = 3 * state[0][6]/80 + 3000
        return Browian_motion_coefficient

    def B_torch(self, state):
        Browian_motion_coefficient = torch.zeros((self.Env.s_dim, self.Env.a_dim)) # 7 x 2
        Browian_motion_coefficient[5, 0] = 3 * state[0][5]/200 + 0.3
        Browian_motion_coefficient[6, 1] = 3 * state[0][6]/80 + 3000
        return Browian_motion_coefficient



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

    def xdot(self, state):

        k10, k20, k30, E1, E2, E3 = self.k10, self.k20, self.k30, self.E1, self.E2, self.E3
        delHRab, delHRbc, delHRad = self.delHRab, self.delHRbc, self.delHRad
        CA0, T0 = self.CA0, self.T0
        rho, Cp, kw, AR, VR = self.rho, self.Cp, self.kw, self.AR, self.VR
        mk, CpK = self.mk, self.CpK

        x = state
        CA, CB, T, TK, VdotVR, QKdot = x

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

        state_no_time = state[0][1: ]

        beta = 10
        alpha = 5
        beta2 = 22
        alpha2 = 0.5
        alpha3 = 70

        m1 = 30.8285163776493
        m2 = 0.1
        kwar = 866.880000000000

        u = np.zeros((2, 1), dtype='float')
        xs = np.array([[0, 0.95, 380, 0, 0, 0]])
        x = state_no_time - xs
        xxs = state_no_time
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




