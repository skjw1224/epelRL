import os
import numpy as np
import casadi as ca
import scipy as sp
import matplotlib.pyplot as plt

from .base_environment import Environment


class CSTR(Environment):
    def __init__(self, config):
        self.env_name = 'CSTR'
        self.config = config

        # Physio-chemical parameters for the CSTR
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

        # Parameters with uncertainty
        self.k10 = 1.287e+12
        self.k20 = 1.287e+12
        self.k30 = 9.043e+9
        self.delHRab = 4.2  # (KJ / MOL)
        self.delHRbc = -11.0  # (KJ / MOL)
        self.delHRad = -41.85  # (KJ / MOL)

        self.param_real = np.array([[self.k10, self.k20, self.k30, self.delHRab, self.delHRbc, self.delHRad]]).T
        self.param_range = np.array([[0.04e12, 0.04e12, 0.27e9, 2.36, 1.92, 1.41]]).T
        self.p_mu = self.param_real
        self.p_sigma = np.zeros([np.shape(self.param_real)[0], 1])
        self.p_eps = np.zeros([np.shape(self.param_real)[0], 1])
        self.param_uncertainty = False
        self.param_extreme = False

        # Dimension
        self.s_dim = 7
        self.a_dim = 2
        self.o_dim = 1
        self.p_dim = np.shape(self.param_real)[0]

        self.t0 = 0.
        self.dt = 30 / 3600.  # hour
        self.tT = 2.  # terminal time

        self.x0 = np.array([[0., 2.1404, 1.4, 387.34, 386.06, 14.19, -1113.5]]).T
        self.u0 = np.array([[0., 0.]]).T
        self.nT = int(self.tT / self.dt)  # episode length

        self.xmin = np.array([[self.t0, 0.001, 0.001, 353.15, 363.15, 3., -9000.]]).T
        self.xmax = np.array([[self.tT, 3.5, 1.8, 413.15, 408.15, 35., 0.]]).T
        self.umin = np.array([[-1., -100.]]).T / self.dt
        self.umax = np.array([[1., 100.]]).T / self.dt
        self.ymin = self.xmin[2]
        self.ymax = self.xmax[2]

        # Basic setup for environment
        self.zero_center_scale = True
        self._set_sym_expressions()
        self.reset()

        self.need_derivs = config['need_derivs']
        self.need_noise_derivs = config['need_noise_derivs']
        self.need_deriv_inverse = config['need_deriv_inverse']

        if self.need_derivs:
            self._eval_model_derivs()

        self.plot_info = {
            'ref_idx_lst': [2],
            'state_plot_shape': (2, 3),
            'action_plot_shape': (1, 2),
            'variable_tag_lst': [
                r'Time[hour]', r'$C_{A}[mol/L]$', r'$C_{B}[mol/L]$', r'$T_{R}[^\circ C]$', r'$T_{C}[^\circ C]$',
                r'$\dot{V}/V_{R}[h^{-1}]$', r'$\dot{Q}[kJ/h]$',
                r'$\Delta\dot{V}/V_{R}[h^{-1}]$', r'$\Delta\dot{Q}[kJ/h]$'
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
            self.p_mu = np.array([[1.327e12, 1.327e12, 8.773e9, 6.56, -9.08, -40.44]]).T
        elif self.param_extreme == 'case2':
            self.p_mu = np.array([[1.247e12, 1.247e12, 9.313e9, 1.84, -12.92, -43.26]]).T
        else:
            self.p_mu = self.param_real

        return x0, u0

    def ref_traj(self):
        return np.array([0.95])
    
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

        # Scaled state & action
        x = np.clip(state, -1, 1)
        u = action
        
        # Identify data_type
        is_term = False
        if self.time_step == self.nT:
            is_term = True

        # Integrate ODE
        if not is_term:
            res = self.I_fnc(x0=x, p=np.concatenate([u, self.p_mu, self.p_sigma, np.random.normal(size=[self.p_dim, 1])]))
            xplus = res['xf'].full()
            cost = res['qf'].full()
            derivs = None

            if self.need_derivs:
                _, dfdx, dfdu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                _, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2 = [_.full() for _ in self.c_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                d2cdu2_inv, Fc, dFcdx, dFcdu = None, None, None, None

                if self.need_deriv_inverse:
                    U = sp.linalg.cholesky(d2cdu2)  # -Huu_inv @ [Hu, Hux, Hus, Hun]
                    d2cdu2_inv = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, np.eye(self.a_dim), lower=True))

                if self.need_noise_derivs:
                    Fc_derivs = [_.full() for _ in self.Fc_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                    Fc = Fc_derivs[0]
                    dFcdx = np.array(Fc_derivs[1:1+self.p_dim])
                    dFcdu = np.array(Fc_derivs[1+self.p_dim:])

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
                    dFcdx = np.array(Fc_derivs[1:1+self.p_dim])
                    dFcdu = np.array(Fc_derivs[1+self.p_dim:])

                derivs = [dfdx, dfdu, dcTdx, dcTdu, d2cTdx2, d2cTdxdu, d2cTdu2, d2cTdu2_inv, Fc, dFcdx, dFcdu]

        noise = np.zeros_like(xplus)
        state_noise = np.random.normal(np.zeros([self.s_dim - self.a_dim - 1, 1]), 0.005*np.ones([self.s_dim - self.a_dim - 1, 1]))
        noise[1:self.s_dim - self.a_dim] = state_noise
        xplus = np.clip(xplus + noise, -1, 1)

        return xplus, cost, is_term, derivs

    def system_functions(self, *args):

        x, u, p_mu, p_sigma, p_eps = args

        x = self.descale(x, self.xmin, self.xmax)
        u = self.descale(u, self.umin, self.umax)

        x = ca.fmax(x, self.xmin)
        u = ca.fmin(ca.fmax(u, self.umin), self.umax)

        k10, k20, k30, E1, E2, E3 = self.k10, self.k20, self.k30, self.E1, self.E2, self.E3
        CA0, T0 = self.CA0, self.T0
        rho, Cp, kw, AR, VR = self.rho, self.Cp, self.kw, self.AR, self.VR
        mk, CpK = self.mk, self.CpK

        # if the variables become 2D array, then use torch.mm()
        p = p_mu + p_eps * p_sigma
        t, CA, CB, T, TK, VdotVR, QKdot = ca.vertsplit(x)
        dVdotVR, dQKdot = ca.vertsplit(u)
        k10, k20, k30, delHRab, delHRbc, delHRad = ca.vertsplit(p)

        k1 = k10 * ca.exp(E1 / T)
        k2 = k20 * ca.exp(E2 / T)
        k3 = k30 * ca.exp(E3 / T)

        dx = [1.,
              VdotVR * (CA0 - CA) - k1 * CA - k3 * CA ** 2.,
              -VdotVR * CB + k1 * CA - k2 * CB,
              VdotVR * (T0 - T) - (k1 * CA * delHRab + k2 * CB * delHRbc + k3 * CA ** 2. * delHRad) /
              (rho * Cp) + (kw * AR) / (rho * Cp * VR) * (TK - T),
              (QKdot + (kw * AR) * (T - TK)) / (mk * CpK),
              dVdotVR,
              dQKdot]

        dx = ca.vertcat(*dx)
        dx = self.scale(dx, self.xmin, self.xmax, shift=False)

        outputs = ca.vertcat(CB)
        y = self.scale(outputs, self.ymin, self.ymax, shift=True)
        return dx, y

    def cost_functions(self, data_type, *args):
        if data_type == 'path':
            x, u, p_mu, p_sigma, p_eps = args  # scaled variable
        else:  # terminal condition
            x, p_mu, p_sigma, p_eps = args  # scaled variable
            u = np.zeros([self.a_dim, 1])

        Q = np.diag([5.])
        R = np.diag([0.1, 0.1])
        H = np.array([0.])

        y = self.y_fnc(x, u, p_mu, p_sigma, p_eps)
        ref = self.scale(self.ref_traj(), self.ymin, self.ymax)

        if data_type == 'path':
            cost = 0.5 * (y - ref).T @ Q @ (y - ref) + 0.5 * u.T @ R @ u
        else:  # terminal condition
            cost = 0.5 * (y - ref).T @ H @ (y - ref)

        return cost
