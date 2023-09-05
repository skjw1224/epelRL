import os
import numpy as np
import casadi as ca
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial


class PfrEnv(object):
    def __init__(self):
        self.env_name = 'PFR'
        self.real_env = False

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
        self.ymin = self.xmin[(1 + 2 * self.space_discretization - self.o_dim):(1 + 3 * self.space_discretization)]
        self.ymax = self.xmax[(1 + 2 * self.space_discretization - self.o_dim):(1 + 3 * self.space_discretization)]

        self.setpoint = 0.20
        # self.setpoint = [0.939, 0.940, 0.941, 0.944, 0.948, 0.95]

        self.zero_center_scale = True

        # MX variable for dae function object (no SX)
        self.state_var = ca.MX.sym('x', self.s_dim)
        self.action_var = ca.MX.sym('u', self.a_dim)
        self.param_mu_var = ca.MX.sym('p_mu', self.p_dim)
        self.param_sigma_var = ca.MX.sym('p_sig', self.p_dim)
        self.param_epsilon_var = ca.MX.sym('p_eps', self.p_dim)

        self.sym_expressions()
        self.dx_derivs, self.Fc_derivs, self.c_derivs, self.cT_derivs = self.eval_model_derivs()

        self.reset()

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
        t0 = self.t0
        u0 = self.scale(self.u0, self.umin, self.umax)

        # Parameter uncertainty
        if self.param_uncertainty:
            self.p_sigma = self.param_range * 0.3
            self.p_eps = np.random.normal(size=[self.p_dim, 1])
        else:
            self.p_sigma = self.param_range * 0
            self.p_eps = np.zeros([self.p_dim, 1])
        self.p_mu = self.param_real

        y0 = self.y_fnc(x0, u0, self.p_mu, self.p_sigma, self.p_eps).full()
        return t0, x0, y0, u0

    def ref_traj(self):
        # ref = 0.145*np.cos(2*np.pi*t) + 0.945 # Cos func btw 1.09 ~ 0.8
        # return np.reshape(ref, [1, -1])
        return np.array([self.setpoint]).reshape([-1, 1])

    def step(self, time, state, action, *args):
        # Scaled state, action, output
        # print("time", time)
        t = round(time, 7)
        x = np.clip(state, -1.03, 1.03)
        u = action

        # Identify data_type
        if t <= self.tT - self.dt:  # leg_BC assigned & interior time --> 'path'
            data_type = 'path'
        elif self.tT - self.dt < t <= self.tT:  # leg BC not assigned & terminal time --> 'terminal'
            data_type = 'terminal'

        # Integrate ODE
        # dx, dxy = self.system_functions(x,u,p_mu,p_sigma,p_eps)
        if data_type == 'path':
            res = self.I_fnc(x0=x, p=np.concatenate([u, self.p_mu, self.p_sigma, np.random.normal(size=[self.p_dim,1])]))
            xplus = res['xf'].full()
            tplus = t + self.dt
            cost = res['qf'].full()
            is_term = False

            _, dfdx, dfdu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
            _, dcdx, _, _, _, d2cdu2 = [_.full() for _ in self.c_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]

            U = sp.linalg.cholesky(d2cdu2)  # -Huu_inv @ [Hu, Hux, Hus, Hun]
            d2cdu2_inv = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, np.eye(self.a_dim), lower=True))
            derivs = [dfdx, dfdu, dcdx, d2cdu2_inv]
        else:
            xplus = x
            tplus = t
            is_term = True

            _, dfdx, dfdu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
            cost, dcTdx, _ = [_.full() for _ in self.cT_derivs(x, self.p_mu, self.p_sigma, self.p_eps)]
            d2cdu2_inv = np.zeros([self.a_dim, self.a_dim])
            derivs = [dfdx, dfdu, dcTdx, d2cdu2_inv]

        # Compute output
        xplus = np.clip(xplus, -1.03, 1.03)
        yplus = self.y_fnc(xplus, u, self.p_mu, self.p_sigma, self.p_eps).full()

        return tplus, xplus, yplus, cost, is_term, derivs

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

    def sym_expressions(self):
        """Syms: :Symbolic expressions, Fncs: Symbolic input/output structures"""

        # lists of sym_vars
        self.path_sym_args = [self.state_var, self.action_var, self.param_mu_var, self.param_sigma_var,
                              self.param_epsilon_var]
        self.term_sym_args = [self.state_var, self.param_mu_var, self.param_sigma_var, self.param_epsilon_var]

        self.path_sym_args_str = ['x', 'u', 'p_mu', 'p_sig', 'p_eps']
        self.term_sym_args_str = ['x', 'p_mu', 'p_sig', 'p_eps']

        "Symbolic functions of f, y"
        self.f_sym, self.y_sym = self.system_functions(*self.path_sym_args)
        self.f_fnc = ca.Function('f_fnc', self.path_sym_args, [self.f_sym], self.path_sym_args_str, ['f'])
        self.y_fnc = ca.Function('y_fnc', self.path_sym_args, [self.y_sym], self.path_sym_args_str, ['y'])

        "Symbolic function of c, cT"
        self.c_sym = partial(self.cost_functions, 'path')(*self.path_sym_args)
        self.cT_sym = partial(self.cost_functions, 'terminal')(*self.term_sym_args)

        self.c_fnc = ca.Function('c_fnc', self.path_sym_args, [self.c_sym], self.path_sym_args_str, ['c'])
        self.cT_fnc = ca.Function('cT_fnc', self.term_sym_args, [self.cT_sym], self.term_sym_args_str, ['cT'])

        "Symbolic function of dae solver"
        dae = {'x': self.state_var,
               'p': ca.vertcat(self.action_var, self.param_mu_var, self.param_sigma_var, self.param_epsilon_var),
               'ode': self.f_sym, 'quad': self.c_sym}
        opts = {'t0': 0., 'tf': self.dt}
        self.I_fnc = ca.integrator('I', 'cvodes', dae, opts)

    def eval_model_derivs(self):
        def ode_state_sensitivity(symargs_path_list):
            state_var, action_var, p_mu_var, p_sigma_var, p_eps_var = symargs_path_list

            ode_p_var = ca.vertcat(action_var, p_mu_var, p_sigma_var, p_eps_var)

            # Jacobian: Adjoint sensitivity
            I_adj = self.I_fnc.factory('I_sym_adj', ['x0', 'p', 'adj:xf', 'adj:qf'], ['adj:x0', 'adj:p'])
            res_sens_xf = I_adj(x0=state_var, p=ode_p_var, adj_xf=np.eye(self.s_dim), adj_qf=0)
            dxfdx = res_sens_xf['adj_x0'].T
            dxfdp = res_sens_xf['adj_p'].T
            dxfdu, dxfdpm, dxfdps, dxfdpe = ca.horzsplit(dxfdp,
                                                         np.cumsum([0, self.a_dim, self.p_dim, self.p_dim, self.p_dim]))

            # dx = fdt + Fc dw
            Fc = dxfdpe / np.sqrt(self.dt)  # SDE correction

            # Taking Jacobian w.r.t ode sensitivity is computationally heavy
            # Instead, compute Hessian of system (d2f/dpedx, d2f/dpedu)
            Fc_direct = ca.jacobian(self.f_sym, self.param_epsilon_var) * np.sqrt(self.dt)

            dFcdx = [ca.jacobian(Fc_direct[:, i], state_var) for i in range(self.p_dim)]
            dFcdu = [ca.jacobian(Fc_direct[:, i], action_var) for i in range(self.p_dim)]

            # Hessian: Forward over adjoint sensitivity (FOA is the most computationally efficient among four methods,
            # i.e., fof, foa, aof, aoa (Automatic Differentiation: Applications, Theory, and Implementations, 239p))
            # I_foa = I_adj.factory('I_foa', ['x0', 'p', 'adj_qf', 'adj_xf', 'fwd:x0', 'fwd:p'], ['fwd:adj_x0', 'fwd:adj_p'])

            # d2xfdx2, d2xfdxu,d2xfdu2 = [], [], []
            # for nxfi in range(self.s_dim):
            #     res_sens_xfx0 = I_foa(x0=state_var, p=action_var, adj_qf=0, adj_xf=np.eye(self.s_dim, 1, k=-nxfi), fwd_x0=np.eye(self.s_dim), fwd_p=0)
            #     d2xfdx2.append(res_sens_xfx0['fwd_adj_x0'].T)
            #     d2xfdxu.append(res_sens_xfx0['fwd_adj_p'].T)
            #
            #     res_sens_xfu0 = I_foa(x0=state_var, p=action_var, adj_qf=0, adj_xf=np.eye(self.s_dim, 1, k=-nxfi), fwd_x0=0, fwd_p=np.eye(self.a_dim))
            #     d2xfdu2.append(res_sens_xfu0['fwd_adj_p'].T)
            # d2xfdx2 = [ca.MX.zeros(self.s_dim, self.s_dim) for _ in range(self.s_dim)]
            # d2xfdxu = [ca.MX.zeros(self.s_dim, self.a_dim) for _ in range(self.s_dim)]
            # d2xfdu2 = [ca.MX.zeros(self.a_dim, self.a_dim) for _ in rca.jacobian(dxfdpe[:, 0], action_var)ange(self.s_dim)]

            # return [dxfdx, dxfdu, *d2xfdx2, *d2xfdxu, *d2xfdu2]

            return dxfdx, dxfdu, dxfdpm, dxfdps, Fc, dFcdx, dFcdu

        def ode_cost_sensitivity(symargs_path_list):
            state_var, action_var, p_mu_var, p_sigma_var, p_eps_var = symargs_path_list

            ode_p_var = ca.vertcat(action_var, p_mu_var, p_sigma_var, p_eps_var)

            # Jacobian: Adjoint sensitivity
            I_adj = self.I_fnc.factory('I_sym_adj', ['x0', 'p', 'adj:xf', 'adj:qf'], ['adj:x0', 'adj:p'])
            res_sens_qf = I_adj(x0=state_var, p=ode_p_var, adj_xf=0, adj_qf=1)
            dcdx = res_sens_qf['adj_x0']
            dcdu = res_sens_qf['adj_p'][:self.a_dim]

            d2cdx2 = ca.jacobian(dcdx, state_var)
            d2cdxu = ca.jacobian(dcdx, action_var)
            d2cdu2 = ca.jacobian(dcdu, action_var)

            # # Hessian: Forward over adjoint sensitivity (FOA is the most computationally efficient among four methods,
            # # i.e., fof, foa, aof, aoa (Automatic Differentiation: Applications, Theory, and Implementations, 239p))
            # I_foa = I_adj.factory('I_foa', ['x0', 'p', 'adj_qf', 'adj_xf', 'fwd:x0', 'fwd:p'],
            #                       ['fwd:adj_x0', 'fwd:adj_p'])
            # res_sens_qfx0 = I_foa(x0=state_var, p=ode_p_var, adj_qf=1, adj_xf=0, fwd_x0=np.eye(self.s_dim), fwd_p=0)
            # d2cdx2 = res_sens_qfx0['fwd_adj_x0'].T
            # d2cdxu = res_sens_qfx0['fwd_adj_p'].T
            # res_sens_qfu0 = I_foa(x0=state_var, p=ode_p_var, adj_qf=1, adj_xf=0, fwd_x0=0, fwd_p=np.eye(self.a_dim))
            # d2cdu2 = res_sens_qfu0['fwd_adj_p'].T

            return [dcdx, dcdu, d2cdx2, d2cdxu, d2cdu2]

        def jac_hess_eval(fnc, x_var, u_var):
            # Compute derivatives of cT, gT, gP, gL, gM
            fnc_dim = fnc.shape[0]

            dfdx = ca.jacobian(fnc, x_var)
            d2fdx2 = [ca.jacobian(dfdx[i, :], x_var) for i in range(fnc_dim)]

            if u_var is None:  # cT, gT
                if fnc_dim == 1:
                    dfdx = dfdx.T
                return [dfdx, *d2fdx2]
            else:  # gP, gL, gM
                dfdu = ca.jacobian(fnc, u_var)
                d2fdxu = [ca.jacobian(dfdx[i, :], u_var) for i in range(fnc_dim)]
                d2fdu2 = [ca.jacobian(dfdu[i, :], u_var) for i in range(fnc_dim)]
                if fnc_dim == 1:
                    dfdx = dfdx.T
                    dfdu = dfdu.T
                return [dfdx, dfdu, *d2fdx2, *d2fdxu, *d2fdu2]

        """f, c: computed from ode sensitivity"""
        dxfdx, dxfdu, dxfdpm, dxfdps, Fc, dFcdx, dFcdu = ode_state_sensitivity(self.path_sym_args)
        f_derivs = ca.Function('f_derivs', self.path_sym_args,
                               [self.f_sym, dxfdx, dxfdu])  # ["F", "Fx", "Fu", "Fxx", "Fxu", "Fuu"]
        Fc_derivs = ca.Function('Fc_derivs', self.path_sym_args, [Fc, *dFcdx, *dFcdu])
        c_derivs = ca.Function('c_derivs', self.path_sym_args, [self.c_sym] + ode_cost_sensitivity(
            self.path_sym_args))  # ["L", "Lx", "Lu", "Lxx", "Lxu", "Luu"]

        """g, cT: computed from pointwise differentiation"""
        # c_derivs = ca.Function('c_derivs', self.path_sym_args, [self.c_sym] + jac_hess_eval(self.c_sym, self.state_var, self.action_var))  # ["L", "Lx", "Lu", "Lxx", "Lxu", "Luu"]
        cT_derivs = ca.Function('cT_derivs', self.term_sym_args,
                                [self.cT_sym] + jac_hess_eval(self.cT_sym, self.state_var,
                                                              None))  # ["LT", "LTx", "LTxx"]

        return f_derivs, Fc_derivs, c_derivs, cT_derivs

    def initial_control(self, i, x):
        if i == 0:
            self.ei = np.zeros([self.s_dim, 1])
        ref = self.scale(self.ref_traj(), self.ymin, self.ymax)
        Kp = 2 * np.ones([self.a_dim, self.s_dim])
        Ki = 0.1 * np.ones([self.a_dim, self.s_dim])
        u = Kp @ (x - ref) + Ki @ self.ei

        self.ei = self.ei + (x - ref)
        return u

    def scale(self, var, min, max, shift=True):
        if self.zero_center_scale == True:  # [min, max] --> [-1, 1]
            shifting_factor = max + min if shift else 0.
            scaled_var = (2. * var - shifting_factor) / (max - min)
        else:  # [min, max] --> [0, 1]
            shifting_factor = min if shift else 0.
            scaled_var = (var - shifting_factor) / (max - min)

        # scaled_var = var

        return scaled_var

    def descale(self, scaled_var, min, max):
        if self.zero_center_scale == True:  # [-1, 1] --> [min, max]
            var = (max - min) / 2 * scaled_var + (max + min) / 2
        else:  # [0, 1] --> [min, max]
            var = (max - min) * scaled_var + min
        #
        # var = scaled_var
        return var

    def tridiagonal(self, a, b, c):
        size = len(self.T0_list) - 2  # -2 to exclude starting & ending points
        B = b * np.eye(size) + a * np.eye(size, k=-1) + c * np.eye(size, k=1)
        B = ca.MX(B)
        return B

    def plot_trajectory(self, traj_data_history, plot_episode, controller_name, save_path):
        variable_tag = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$',
                        r'$C_{A,1}$', r'$C_{A,2}$', r'$C_{A,3}$', r'$C_{A,4}$', r'$C_{A,5}$', r'$C_{A,6}$',
                        r'$T_{W,1}$', r'$T_{W,2}$', r'$\Delta T_{W,1}$', r'$\Delta T_{W,2}$']
        time = traj_data_history[0, :, 0]
        ref = traj_data_history[0, :, -1]
        num_saved_epi = traj_data_history.shape[0]

        fig1, ax1 = plt.subplots(nrows=3, ncols=6, figsize=(20, 12))
        fig1.subplots_adjust(hspace=.4, wspace=.5)
        ax1.flat[11].plot(time, ref, 'r--', label='Set point')
        for i in range(self.s_dim + self.a_dim - 1):
            ax1.flat[i].set_xlabel(r'time')
            ax1.flat[i].set_ylabel(variable_tag[i])
            for j in range(num_saved_epi):
                epi = plot_episode[j]
                ax1.flat[i].plot(time, traj_data_history[j, :, i+1], label=f'Episode {epi}')
            ax1.flat[i].legend()
            ax1.flat[i].grid()
        fig1.tight_layout()
        plt.savefig(os.path.join(save_path, f'{self.env_name}_{controller_name}_var_traj.png'))

        fig2, ax2 = plt.subplots(figsize=(10,6))
        ax2.plot(time, ref, 'r--', label='Set point')
        ax2.set_xlabel(r'time')
        ax2.set_ylabel(variable_tag[11])
        for j in range(num_saved_epi):
            epi = plot_episode[j]
            ax2.plot(time, traj_data_history[j, :, 12], label=f'Episode {epi}')
        ax2.legend()
        ax2.grid()
        fig2.tight_layout()
        plt.savefig(os.path.join(save_path, f'{self.env_name}_{controller_name}_CV_traj.png'))
        plt.show()
