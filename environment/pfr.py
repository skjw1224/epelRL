import os
import numpy as np
import casadi as ca
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial

from .base_environment import Environment


class PFR(Environment):
    def __init__(self, config):
        self.env_name = 'PFR'
        self.config = config

        # Physio-chemical parameters
        self.Peh = 5.
        self.Pem = 5.
        self.Le = 1.
        self.Da = 0.875
        self.gamma = 15.
        self.eta = 0.8375
        self.mu = 13.
        self.Tw0 = 1.
        self.T0 = 1.
        self.CA0 = 1.

        self.param_real = np.array(
            [[self.Peh, self.Pem, self.Le, self.Da, self.gamma, self.eta, self.mu, self.Tw0, self.T0, self.CA0]]).T
        self.param_range = self.param_real * 0.1
        self.p_mu = self.param_real
        self.p_sigma = np.zeros([np.shape(self.param_real)[0], 1])
        self.p_eps = np.zeros([np.shape(self.param_real)[0], 1])
        self.param_uncertainty = False
        self.param_extreme = False

        self.T0_list = [.6, .4, .3, .3, .3, .3]
        self.CA0_list = [1., 1., 1., 1., 1., 1.]
        self.Tw0_list = [1., 1.]
        self.space_discretization = len(self.T0_list)

        self.Tmin_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.CAmin_list = [0., 0., 0., 0., 0., 0.]
        self.Twmin_list = [0.3, 0.3]

        self.Tmax_list = [2., 2., 2., 2., 2., 2.]
        self.CAmax_list = [1., 1., 1., 1., 1., 1.]
        self.Twmax_list = [2., 2.]

        self.s_dim = len(self.T0_list) + len(self.CA0_list) + len(self.Tw0_list) + 1
        self.a_dim = len(self.Tw0_list)
        self.o_dim = 1
        self.p_dim = np.shape(self.param_real)[0]

        self.t0 = 0.
        self.dt = 20 / 3600.  # hour
        self.tT = 3600 / 3600.  # terminal time

        self.x0_list = np.hstack(([0.], self.T0_list, self.CA0_list, self.Tw0_list))
        self.xmin_list = np.hstack(([self.t0], self.Tmin_list, self.CAmin_list, self.Twmin_list))
        self.xmax_list = np.hstack(([self.tT], self.Tmax_list, self.CAmax_list, self.Twmax_list))

        self.x0 = self.x0_list.reshape(-1, 1)
        self.u0 = np.array([[0., 0.]]).T
        self.nT = int(self.tT / self.dt) + 1  # episode length

        self.xmin = self.xmin_list.reshape((-1, 1))
        self.xmax = self.xmax_list.reshape((-1, 1))
        self.umin = np.array([[-0.05, -0.05]]).T / self.dt
        self.umax = np.array([[0.05, 0.05]]).T / self.dt
        self.ymin = self.xmin[(1 + 2 * self.space_discretization - self.o_dim):(1 + 2 * self.space_discretization)]
        self.ymax = self.xmax[(1 + 2 * self.space_discretization - self.o_dim):(1 + 2 * self.space_discretization)]

        self.setpoint = 0.20
        # self.setpoint = [0.939, 0.940, 0.941, 0.944, 0.948, 0.95]

        self.zero_center_scale = True
        self._set_sym_expressions()
        self.reset()

        self.need_derivs = config.need_derivs
        self.need_noise_derivs = config.need_noise_derivs
        self.need_deriv_inverse = config.need_deriv_inverse

        if self.need_derivs:
            self._eval_model_derivs()

        self.plot_info = {
            'ref_idx_lst': [12],
            'state_plot_shape': (3, 5),
            'action_plot_shape': (1, 2),
            'variable_tag_lst': ['time',
                r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$',
                r'$C_{A,1}$', r'$C_{A,2}$', r'$C_{A,3}$', r'$C_{A,4}$', r'$C_{A,5}$', r'$C_{A,6}$',
                r'$T_{W,1}$', r'$T_{W,2}$', r'$\Delta T_{W,1}$', r'$\Delta T_{W,2}$'
            ]
        }


    def reset(self, x0=None, random_init=False):
        if x0 is None:
            if random_init == False:
                x0 = self.x0
            else:
                x0 = self.x0
                # t, u0 should not be initialized randomly
                x0[1:self.s_dim - 2] = self.descale(np.random.uniform(-0.3, 0.3, [self.s_dim - 3, 1]),
                                                    self.xmin[1:self.s_dim - 2], self.xmax[1:self.s_dim - 2])

        x0 = self.scale(x0, self.xmin, self.xmax)
        u0 = self.scale(self.u0, self.umin, self.umax)

        self.time_step = 0

        # Parameter uncertainty
        if self.param_uncertainty:
            self.p_sigma = self.param_range * 0.3
            self.p_eps = np.random.normal(size=[self.p_dim, 1])
        else:
            self.p_sigma = self.param_range * 0
            self.p_eps = np.zeros([self.p_dim, 1])
        self.p_mu = self.param_real
        return x0, u0

    def ref_traj(self):
        # ref = 0.145*np.cos(2*np.pi*t) + 0.945 # Cos func btw 1.09 ~ 0.8
        # return np.reshape(ref, [1, -1])
        return np.array([self.setpoint]).reshape([-1, 1])

    def pid_gain(self):
        Kp = 2.0 * np.ones((self.a_dim, self.o_dim))
        Ki = 0.1 * np.ones((self.a_dim, self.o_dim))
        Kd = np.zeros((self.a_dim, self.o_dim))

        return {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}

    def get_observ(self, state, action):
        observ = self.y_fnc(state, action, self.p_mu, self.p_sigma, self.p_eps).full()

        return observ

    def step(self, state, action):
        self.time_step += 1

        x = np.clip(state, -1.03, 1.03)
        u = action

        # Identify data_type
        is_term = False
        if self.time_step == self.nT:
            is_term = True

        # Integrate ODE
        if not is_term:
            res = self.I_fnc(x0=x, p=np.concatenate([u, self.p_mu, self.p_sigma, np.random.normal(size=[self.p_dim,1])]))
            xplus = res['xf'].full()
            cost = res['qf'].full()
            derivs = None

            if self.need_derivs:
                _, dfdx, dfdu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                _, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2 = [_.full() for _ in
                                                          self.c_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                d2cdu2_inv, Fc, dFcdx, dFcdu = None, None, None, None

                if self.need_deriv_inverse:
                    U = sp.linalg.cholesky(d2cdu2)  # -Huu_inv @ [Hu, Hux, Hus, Hun]
                    d2cdu2_inv = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, np.eye(self.a_dim),
                                                                                          lower=True))

                if self.need_noise_derivs:
                    Fc_derivs = [_.full() for _ in self.Fc_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                    Fc = Fc_derivs[0]
                    dFcdx = np.array(Fc_derivs[1:1 + self.p_dim])
                    dFcdu = np.array(Fc_derivs[1 + self.p_dim:])

                derivs = [dfdx, dfdu, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2, d2cdu2_inv, Fc, dFcdx, dFcdu]
        else:
            xplus = x
            cost = self.cT_fnc(x, self.p_mu, self.p_sigma, self.p_eps).full()
            derivs = None

            if self.need_derivs:
                _, dfdx, dfdu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                _, dcTdx, d2cTdx2 = [_.full() for _ in self.cT_derivs(x, self.p_mu, self.p_sigma, self.p_eps)]
                dcTdu = np.zeros([self.a_dim, 1])
                d2cTdxdu = np.zeros([self.s_dim, self.a_dim])
                d2cTdu2 = np.zeros([self.a_dim, self.a_dim])
                d2cTdu2_inv = np.zeros([self.a_dim, self.a_dim])

                Fc, dFcdx, dFcdu = None, None, None

                if self.need_noise_derivs:
                    Fc_derivs = self.Fc_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)
                    Fc = Fc_derivs[0]
                    dFcdx = np.array(Fc_derivs[1:1 + self.p_dim])
                    dFcdu = np.array(Fc_derivs[1 + self.p_dim:])

                derivs = [dfdx, dfdu, dcTdx, dcTdu, d2cTdx2, d2cTdxdu, d2cTdu2, d2cTdu2_inv, Fc, dFcdx, dFcdu]

        noise = np.zeros_like(xplus)
        state_noise = np.random.normal(np.zeros([self.s_dim - self.a_dim - 1, 1]),
                                       0.005 * np.ones([self.s_dim - self.a_dim - 1, 1]))
        noise[1:self.s_dim - self.a_dim] = state_noise
        xplus = np.clip(xplus + noise, -1, 1)

        return xplus, cost, is_term, derivs

    def system_functions(self, *args):

        x, u, p_mu, p_sigma, p_eps = args

        x = self.descale(x, self.xmin, self.xmax)
        u = self.descale(u, self.umin, self.umax)

        x = ca.fmax(x, self.xmin)
        u = ca.fmin(ca.fmax(u, self.umin), self.umax)

        # if the variables become 2D array, then use torch.mm()
        p = p_mu + p_eps * p_sigma
        t, T1, T2, T3, T4, T5, T6, CA1, CA2, CA3, CA4, CA5, CA6, Tw1, Tw2 = ca.vertsplit(x)  # CB1,...
        dTw1, dTw2 = ca.vertsplit(u)
        Peh, Pem, Le, Da, gamma, eta, mu, Tw0, T0, CA0 = ca.vertsplit(p)
        deltaz = 1 / (len(self.T0_list) - 1)

        R1 = CA1 * ca.exp(gamma * (1 - 1 / T1))
        R2 = CA2 * ca.exp(gamma * (1 - 1 / T2))
        R3 = CA3 * ca.exp(gamma * (1 - 1 / T3))
        R4 = CA4 * ca.exp(gamma * (1 - 1 / T4))
        R5 = CA5 * ca.exp(gamma * (1 - 1 / T5))
        R6 = CA6 * ca.exp(gamma * (1 - 1 / T6))

        # dT1 = -Peh * (T0 - T1) - (-Peh/Le) * (T0 - T1) + eta * R1 + mu * (Tw1 - T1)
        dT1 = (1 / Peh) * (2 * T2 - 2 * T1) / (deltaz ** 2) + (2 / deltaz + Peh / Le) * (T0 - T1) + eta * R1 + mu * (
                    Tw1 - T1)
        dT2 = (1 / Peh) * (T1 - 2 * T2 + T3) / (deltaz ** 2) - (1 / Le) * (-T1 + T3) / (2 * deltaz) + eta * R2 + mu * (
                    Tw1 - T2)
        dT3 = (1 / Peh) * (T2 - 2 * T3 + T4) / (deltaz ** 2) - (1 / Le) * (-T2 + T4) / (2 * deltaz) + eta * R3 + mu * (
                    Tw1 - T3)
        dT4 = (1 / Peh) * (T3 - 2 * T4 + T5) / (deltaz ** 2) - (1 / Le) * (-T3 + T5) / (2 * deltaz) + eta * R4 + mu * (
                    Tw2 - T4)
        dT5 = (1 / Peh) * (T4 - 2 * T5 + T6) / (deltaz ** 2) - (1 / Le) * (-T4 + T6) / (2 * deltaz) + eta * R5 + mu * (
                    Tw2 - T5)
        dT6 = (1 / Peh) * (2 * T5 - 2 * T6) / (deltaz ** 2) + eta * R6 + mu * (Tw2 - T6)

        # dCA1 = -Pem * (CA0 - CA1) + Pem * (CA0 - CA1) - Da * R1
        dCA1 = (1 / Pem) * (2 * CA2 - 2 * CA1) / (deltaz ** 2) + (2 / deltaz + Pem) * (CA0 - CA1) - Da * R1
        dCA2 = (1 / Pem) * (CA1 - 2 * CA2 + CA3) / (deltaz ** 2) - (-CA1 + CA3) / (2 * deltaz) - Da * R2
        dCA3 = (1 / Pem) * (CA2 - 2 * CA3 + CA4) / (deltaz ** 2) - (-CA2 + CA4) / (2 * deltaz) - Da * R3
        dCA4 = (1 / Pem) * (CA3 - 2 * CA4 + CA5) / (deltaz ** 2) - (-CA3 + CA5) / (2 * deltaz) - Da * R4
        dCA5 = (1 / Pem) * (CA4 - 2 * CA5 + CA6) / (deltaz ** 2) - (-CA4 + CA6) / (2 * deltaz) - Da * R5
        dCA6 = (1 / Pem) * (2 * CA5 - 2 * CA6) / (deltaz ** 2) - Da * R6

        dx = [1.,
              dT1, dT2, dT3, dT4, dT5, dT6,
              dCA1, dCA2, dCA3, dCA4, dCA5, dCA6,
              # -1*dCA1, -1*dCA2, -1*dCA3, -1*dCA4, -1*dCA5, -1*dCA6,
              dTw1, dTw2]

        dx = ca.vertcat(*dx)
        dx = self.scale(dx, self.xmin, self.xmax, shift=False)

        outputs = ca.vertcat(CA6)
        y = self.scale(outputs, self.ymin, self.ymax, shift=True)
        return dx, y

    def cost_functions(self, data_type, *args):
        if data_type == 'path':
            x, u, p_mu, p_sigma, p_eps = args  # scaled variable
        else:  # terminal condition
            x, p_mu, p_sigma, p_eps = args  # scaled variable
            u = np.zeros([self.a_dim, 1])

        Q = np.eye(self.o_dim) * 3.0  # np.diag([3.])
        R = np.diag([0.1, 0.1])
        H = np.array([0.])

        y = self.y_fnc(x, u, p_mu, p_sigma, p_eps)
        ref = self.scale(self.ref_traj(), self.ymin, self.ymax)

        if data_type == 'path':
            cost = 0.5 * (y - ref).T @ Q @ (y - ref) + 0.5 * u.T @ R @ u
        else:  # terminal condition
            cost = 0.5 * (y - ref).T @ H @ (y - ref)

        # w1 = 0.5 * (y - ref).T @ Q @ (y - ref)
        # w2 = 0.5 * u.T @ R @ u
        return cost

    def initial_control(self, i, x):
        if i == 0:
            self.ei = np.zeros([self.s_dim, 1])
        ref = self.scale(self.ref_traj(), self.ymin, self.ymax)
        Kp = 2 * np.ones([self.a_dim, self.s_dim])
        Ki = 0.1 * np.ones([self.a_dim, self.s_dim])
        u = Kp @ (x - ref) + Ki @ self.ei

        self.ei = self.ei + (x - ref)
        return u

    def tridiagonal(self, a, b, c):
        size = len(self.T0_list) - 2  # -2 to exclude starting & ending points
        B = b * np.eye(size) + a * np.eye(size, k=-1) + c * np.eye(size, k=1)
        B = ca.MX(B)
        return B