import os
import numpy as np
import casadi as ca
import scipy as sp
import matplotlib.pyplot as plt

from .base_environment import Environment

# Physio-chemical parameters for the crystallization reactor
Cs = 0.46           # saturation concentration [kg/kg]
g = 1               # growth rate exponent [-]
Hc = 60.75          # specific enthalpy of crystals [kJ/kg]
Hl = 69.86          # specific enthalpy of liquid [kJ/kg]
Hv = 2.59E3         # specific enthalpy of vapor [kJ/kg]
Kv = 0.43           # volumetric shape factor
kb = 1.02E14        # nucleation rate constant [#/m4]
kg = 7.5E-5         # growth rate constant [m/s]
Fp = 1.73E-6        # product flow rate [m3/s]
V = 7.5E-2          # crystallizer volume [m3]
rhoc = 1767.35      # crystal density [kg/m3]
rhol = 1248.93      # solutio density [kg/m3]

# Parameters with uncertainty
Kv = 0.43           # volumetric shape factor

class CRYSTAL(Environment):
    def __init__(self, config):
        self.env_name = 'CRYSTAL'
        self.config = config

        # Uncertain parameters: delH_R, k_0
        self.param_real = np.array([[Kv]]).T
        self.param_range = np.array([[Kv * 0.3]]).T
        self.p_mu = self.param_real
        self.p_sigma = np.zeros([np.shape(self.param_real)[0], 1])
        self.p_eps = np.zeros([np.shape(self.param_real)[0], 1])
        self.param_uncertainty = False
        self.param_extreme = False

        # Dimension
        self.s_dim = 8
        self.a_dim = 1
        self.o_dim = 1
        self.p_dim = np.shape(self.param_real)[0]

        self.t0 = 0.
        self.dt = 10
        self.tT = 10000.

        # state: t, m0, m1, m2, m3, m4, C, Qv
        # action: dQv
        # observation: m_P

        m0i, m1i, m2i, m3i, m4i = [1E11, 4E6, 400, 0.1, 0.2E-4]
        C0 = 0.461
        Qv0 = 8.5  # kW

        self.x0 = np.array([[self.t0, m0i, m1i, m2i, m3i, m4i, C0, Qv0]]).T
        self.u0 = np.array([[0.]]).T
        self.nT = int(self.tT / self.dt)  # episode length

        self.xmin = np.array([[self.t0, m0i, m1i, m2i, m3i, m4i, Cs, 7]]).T
        self.xmax = np.array([[self.tT, m0i * 1.2, m1i * 10, m2i * 10, m3i *10, m4i*10, C0, 20]]).T
        self.umin = np.array([[-0.005]]).T
        self.umax = np.array([[0.005]]).T
        self.ymin = np.array([[m3i / m2i - (m3i / m2i) / 2]])
        self.ymax = np.array([[m3i / m2i]])

        # Basic setup for environment
        self.zero_center_scale = False
        self._set_sym_expressions()
        self.reset()

        self.need_derivs = config.need_derivs
        self.need_noise_derivs = config.need_noise_derivs
        self.need_deriv_inverse = config.need_deriv_inverse

        if self.need_derivs:
            self._eval_model_derivs()

        self.plot_info = {
            'ref_idx_lst': [],
            'state_plot_shape': (2, 4),
            'action_plot_shape': (1, 1),
            'variable_tag_lst': [
                r'Time[s]', r'$m_{0}[#]$', r'$m_{1}[m]$', r'$m_{2}[m^2]$',
                r'$m_{3}[m^3]$', r'$m_{4}[m^4]$', r'$C[kg/kg]$', r'$Q[kW]$',
                r'$\Delta Q[kW]$'
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
            self.p_mu = np.array([[Kv * 0.70]]).T
        elif self.param_extreme == 'case2':
            self.p_mu = np.array([[Kv * 1.30]]).T
        else:
            self.p_mu = self.param_real

        return x0, u0

    def ref_traj(self):
        return np.zeros([self.o_dim, ])

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
            x = np.clip(state, -2, 2)
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
        t, m0, m1, m2, m3, m4, C, Qv = ca.vertsplit(x)
        dQv = ca.vertsplit(u)[0]
        Kv = ca.vertsplit(p)[0]

        k1 = Hv * Cs / (Hv - Hl) * (rhoc / rhol - 1 + (rhol * Hl - rhoc * Hc) / (rhol * Hv)) - rhoc / rhol
        k2 = Cs / (V * rhol * (Hv - Hl))
        G = kg * (C - Cs) ** g
        B0 = kb * m3 * G

        dt = 1.
        dm0 = B0 - m0 * Fp / V
        dm1 = G * m0 - m1 * Fp / V
        dm2 = 2 * G * m1 - m2 * Fp / V
        dm3 = 3 * G * m2 - m3 * Fp / V
        dm4 = 4 * G * m3 - m4 * Fp / V
        dC = (Fp * (Cs - C) / V + 3 * Kv * G * m2 * (k1 + C)) / (1 - Kv * m3) + k2 * Qv / (1 - Kv * m3)
        dQv = dQv

        dx = [dt, dm0, dm1, dm2, dm3, dm4, dC, dQv]

        dx = ca.vertcat(*dx)
        dx = self.scale(dx, self.xmin, self.xmax, shift=False)

        outputs = ca.vertcat(dm2, dm3, Qv)
        y = self.scale(outputs, self.ymin, self.ymax, shift=True)
        return dx, y

    def cost_functions(self, data_type, *args):
        if data_type == 'path':
            x, u, p_mu, p_sigma, p_eps = args  # scaled variable
        else:  # terminal condition
            x, p_mu, p_sigma, p_eps = args  # scaled variable
            u = np.zeros([self.a_dim, 1])

        Q = np.diag([1.]) * 0.0001
        R = np.diag([1.]) * 0.0001
        H = np.diag([1.]) * 10
        uref = np.diag([0.5])

        y = self.y_fnc(x, u, p_mu, p_sigma, p_eps)
        y = self.descale(y, self.ymin, self.ymax)
        dm2, dm3, Qv = ca.vertsplit(y)

        if data_type == 'path':
            cost = Qv ** 2 * Q + (u - uref) @ R @ (u - uref).T
        else:  # terminal condition
            cost = -dm3 / dm2 @ H

        return cost
