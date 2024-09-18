import os
import numpy as np
import casadi as ca
import scipy as sp
import matplotlib.pyplot as plt

from .base_environment import Environment

# Physio-chemical parameters for the CSTR
feed = 24.0 / 60.0 # Feed flow rate [mol/hr]
xf = 0.5    # Feed mole fraction
mtr = 0.25  # Tray holdup [mol]
mcd = 0.5   # Condenser holdup [mol]
mrb = 1.0   # Reboiler holdup [mol]
D = feed* xf # Distillate flow rate

alpha = 1.6 # relative volatility

NTRAYS = 32 # Number of trays
S_FEED = 16 # Feed stage
S_RECT = range(1, S_FEED) # Rectification section stage numbers
S_STRIP = range(S_FEED + 1, NTRAYS - 1) # Stripping section stage numbers

# Parameters with uncertainty
alpha = 1.6

class DISTILLATION(Environment):
    def __init__(self, config):
        self.env_name = 'DISTILLATION'
        self.config = config

        self.param_real = np.array([[alpha]]).T
        self.param_range = np.array([[alpha * 0.3]]).T
        self.p_mu = self.param_real
        self.p_sigma = np.zeros([np.shape(self.param_real)[0], 1])
        self.p_eps = np.zeros([np.shape(self.param_real)[0], 1])
        self.param_uncertainty = False
        self.param_extreme = False

        # Dimension
        self.s_dim = NTRAYS + 2
        self.a_dim = 1
        self.o_dim = 2
        self.p_dim = np.shape(self.param_real)[0]

        self.t0 = 0.
        self.dt = 1/4.
        self.tT = 100.

        # state: t, x (for all trays), rr
        # action: drr
        # observation: x[Distillate], rr

        x_tray0 = np.array([0.935419416, 0.900525537, 0.862296451, 0.821699403,
           0.779990796, 0.738571686, 0.698804909, 0.661842534, 0.628507776, 0.5992527,
           0.57418568, 0.553144227, 0.535784544, 0.52166551, 0.510314951, 0.501275092,
           0.494128917, 0.48544992, 0.474202481, 0.459803499, 0.441642973, 0.419191098,
           0.392055492, 0.360245926, 0.32407993, 0.284676816, 0.243209213, 0.201815683,
           0.16177269, 0.12514971, 0.092458326, 0.064583177])

        rr0 = 3.0

        self.x0 = np.array([[self.t0, *x_tray0, rr0]]).T
        self.u0 = np.array([[0.]]).T
        self.nT = int(self.tT / self.dt)  # episode length

        self.xmin = np.array([[self.t0, *np.zeros(NTRAYS, ), 1]]).T
        self.xmax = np.array([[self.tT, *np.ones(NTRAYS, ), 5]]).T
        self.umin = np.array([[-0.5]]).T
        self.umax = np.array([[0.5]]).T
        self.ymin = np.array([[0, 1]]).T
        self.ymax = np.array([[1, 5]]).T

        # Basic setup for environment
        self.zero_center_scale = True
        self._set_sym_expressions()
        self.reset()

        self.need_derivs = config.need_derivs
        self.need_noise_derivs = config.need_noise_derivs
        self.need_deriv_inverse = config.need_deriv_inverse

        if self.need_derivs:
            self._eval_model_derivs()

        self.plot_info = {
            'ref_idx_lst': [1, 6],
            'state_plot_idx_lst': [1, int(NTRAYS / 4), S_FEED, int(NTRAYS / 4 * 3), NTRAYS, NTRAYS + 1],
            'state_plot_shape': (2, 3),
            'action_plot_shape': (1, 1),
            'variable_tag_lst': [
                r'Time[hour]',
                r'$x_{Distillate}[-]$', r'$x_{Rectification}[-]$', r'$x_{Feed}[-]$', r'$x_{Stripping}[-]$', r'$x_{Bottom}[-]$',
                r'$Reflux ratio[-]$',
                r'$\Delta Reflux ratio[s^{-1}]$'
            ]
        }

    def reset(self, x0=None, random_init=False):
        if x0 is None:
            if random_init == False:
                x0 = self.x0
            else:
                x0 = self.x0
                # t, u0 should not be initialized randomly
                x0[1:self.s_dim - 1] = self.descale(np.random.uniform(-0.3, 0.3,
                                                                      [self.s_dim - self.a_dim - 1, 1]),
                                                    self.xmin[1:self.s_dim - 1], self.xmax[1:self.s_dim - 1])

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

        if self.param_extreme == 'case1':
            self.p_mu = np.array([[alpha * 0.70]]).T
        elif self.param_extreme == 'case2':
            self.p_mu = np.array([[alpha * 1.30]]).T
        else:
            self.p_mu = self.param_real

        return x0, u0

    def ref_traj(self):
        return np.array([[0.895814, 2.0]]).T

    def gain(self):
        Kp = 2.0 * np.ones((self.a_dim, self.o_dim))
        Ki = 0.1 * np.ones((self.a_dim, self.o_dim))
        Kd = np.zeros((self.a_dim, self.o_dim))

        return {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}

    def get_observ(self, state, action):
        observ = self.y_fnc(state, action, self.p_mu, self.p_sigma, self.p_eps).full()

        return observ

    def step(self, state, action):
        self.time_step += 1

        # Scaled state & action
        if self.zero_center_scale:
            x = np.clip(state, -1, 1)
        else:
            x = np.clip(state, 0, 2)
        u = action

        # Identify data_type
        is_term = False
        if self.time_step == self.nT:
            is_term = True

        # Integrate ODE
        if not is_term:
            res = self.I_fnc(x0=x,
                             p=np.concatenate([u, self.p_mu, self.p_sigma, np.random.normal(size=[self.p_dim, 1])]))
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
        if self.zero_center_scale:
            xplus = np.clip(xplus + noise, -2, 2)
        else:
            xplus = np.clip(xplus + noise, 0, 2)

        return xplus, cost, is_term, derivs

    def system_functions(self, *args):

        x, u, p_mu, p_sigma, p_eps = args

        x = self.descale(x, self.xmin, self.xmax)
        u = self.descale(u, self.umin, self.umax)

        x = ca.fmax(x, self.xmin)
        u = ca.fmin(ca.fmax(u, self.umin), self.umax)

        # if the variables become 2D array, then use torch.mm()
        p = p_mu + p_eps * p_sigma
        t = x[0]
        xl = x[1:-1]
        rr = x[-1]

        drr = ca.vertsplit(u)[0]
        alpha = ca.vertsplit(p)[0]

        L = rr * D
        V = L + D
        FL = feed + L

        yl = xl * alpha / (1 + (alpha - 1) * xl)
        dxl = []

        dxl0 = 1 / mcd * V * (yl[1] - xl[0])
        dxl.append(dxl0)

        for n in S_RECT:
            dxln = 1 / mtr * (L * (xl[n - 1] - xl[n]) - V * (yl[n] - yl[n + 1]))
            dxl.append(dxln)

        dxlf = 1 / mtr * (feed * xf + L * xl[S_FEED - 1] - FL * xl[S_FEED] - V * (yl[S_FEED] - yl[S_FEED + 1]))
        dxl.append(dxlf)

        for n in S_STRIP:
            dxln = 1 / mtr * (FL * (xl[n - 1] - xl[n]) - V * (yl[n] - yl[n + 1]))
            dxl.append(dxln)

        dxlb = 1 / mrb * (FL * xl[NTRAYS - 2] - (feed - D) * xl[NTRAYS - 1] - V * yl[NTRAYS - 1])
        dxl.append(dxlb)

        dt = 1.
        drr = drr

        dx = [dt, *dxl, drr]

        dx = ca.vertcat(*dx)
        dx = self.scale(dx, self.xmin, self.xmax, shift=False)

        outputs = ca.vertcat(xl[0], rr)
        y = self.scale(outputs, self.ymin, self.ymax, shift=True)
        return dx, y

    def cost_functions(self, data_type, *args):
        if data_type == 'path':
            x, u, p_mu, p_sigma, p_eps = args  # scaled variable
        else:  # terminal condition
            x, p_mu, p_sigma, p_eps = args  # scaled variable
            u = np.zeros([self.a_dim, 1])

        Q = np.diag([5., 0.05])
        R = np.diag([0.01])
        H = np.array([0.])

        y = self.y_fnc(x, u, p_mu, p_sigma, p_eps)
        ref = self.scale(self.ref_traj(), self.ymin, self.ymax)

        if data_type == 'path':
            cost = 0.5 * (y - ref).T @ Q @ (y - ref) + 0.5 * u.T @ R @ u
        else:  # terminal condition
            cost = 0.5 * (y - ref).T @ H @ (y - ref)

        return cost
