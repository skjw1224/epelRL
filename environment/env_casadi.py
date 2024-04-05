import os
import numpy as np
import casadi as ca
import scipy as sp
import matplotlib.pyplot as plt
from functools import partial
from .base_environment import Environment


class CSTR(Environment):
    def __init__(self):
        self.env_name = 'CSTR'
        self.real_env = False

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
        self.dt = 20 / 3600.  # hour
        self.tT = 3600 / 3600.  # terminal time

        self.x0 = np.array([[0., 2.1404, 1.4, 387.34, 386.06, 14.19, -1113.5]]).T
        self.u0 = np.array([[0., 0.]]).T
        self.nT = int(self.tT / self.dt)  # episode length

        self.xmin = np.array([[self.t0, 0.001, 0.001, 353.15, 363.15, 3., -9000.]]).T
        self.xmax = np.array([[self.tT, 3.5, 1.8, 413.15, 408.15, 35., 0.]]).T
        self.umin = np.array([[-1., -1000.]]).T / self.dt
        self.umax = np.array([[1., 1000.]]).T / self.dt
        self.ymin = self.xmin[2]
        self.ymax = self.xmax[2]

        self.zero_center_scale = True

        # MX variables for dae function object (no SX)
        self.state_var = ca.MX.sym('x', self.s_dim)
        self.action_var = ca.MX.sym('u', self.a_dim)
        self.param_mu_var = ca.MX.sym('p_mu', self.p_dim)
        self.param_sigma_var = ca.MX.sym('p_sig', self.p_dim)
        self.param_epsilon_var = ca.MX.sym('p_eps', self.p_dim)

        self.sym_expressions()
        self.dx_derivs, self.Fc_derivs, self.c_derivs, self.cT_derivs = self.eval_model_derivs()

        self.reset()

        self.plot_info = {
            'ref_idx_lst': [2],
            'state_plot_shape': (2, 3),
            'action_plot_shape': (1, 2),
            'ref_idx_lst': [2],
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
                x0[1:5] = self.descale(np.random.uniform(-0.3, 0.3, [4, 1]), self.xmin[1:5], self.xmax[1:5])

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

        if self.param_extreme == 'case1':
            self.p_mu = np.array([[1.327e12, 1.327e12, 8.773e9, 6.56, -9.08, -40.44]]).T
        elif self.param_extreme == 'case2':
            self.p_mu = np.array([[1.247e12, 1.247e12, 9.313e9, 1.84, -12.92, -43.26]]).T
        else:
            self.p_mu = self.param_real

        y0 = self.y_fnc(x0, u0, self.p_mu, self.p_sigma, self.p_eps).full()
        return t0, x0, y0, u0

    def ref_traj(self):
        return np.array([0.95])

    def step(self, time, state, action, *args):
        # Scaled state, action, output
        t = round(time, 7)
        x = np.clip(state, -2, 2)
        u = action

        # Identify data_type
        is_term = False
        if self.tT - self.dt < t <= self.tT:  # leg BC not assigned & terminal time --> 'terminal'
            is_term = True

        # Integrate ODE
        if not is_term:
            res = self.I_fnc(x0=x, p=np.concatenate([u, self.p_mu, self.p_sigma, np.random.normal(size=[self.p_dim, 1])]))
            xplus = res['xf'].full()
            tplus = t + self.dt
            cost = res['qf'].full()

            _, dfdx, dfdu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
            _, dcdx, _, _, _, d2cdu2 = [_.full() for _ in self.c_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]

            U = sp.linalg.cholesky(d2cdu2)  # -Huu_inv @ [Hu, Hux, Hus, Hun]
            d2cdu2_inv = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, np.eye(self.a_dim), lower=True))
            derivs = [dfdx, dfdu, dcdx, d2cdu2_inv]
        else:
            xplus = x
            tplus = t

            _, dfdx, dfdu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
            cost, dcTdx, _ = [_.full() for _ in self.cT_derivs(x, self.p_mu, self.p_sigma, self.p_eps)]
            d2cdu2_inv = np.zeros([self.a_dim, self.a_dim])
            derivs = [dfdx, dfdu, dcTdx, d2cdu2_inv]

        # Compute output
        xplus = np.clip(xplus, -2, 2)
        yplus = self.y_fnc(xplus, u, self.p_mu, self.p_sigma, self.p_eps).full()

        return tplus, xplus, yplus, cost, is_term, derivs

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

    def sym_expressions(self):
        """Syms: Symbolic expressions, Fncs: Symbolic input/output structures"""

        # lists of sym_vars
        self.path_sym_args = [self.state_var, self.action_var, self.param_mu_var, self.param_sigma_var, self.param_epsilon_var]
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
        dae = {'x': self.state_var, 'p': ca.vertcat(self.action_var, self.param_mu_var, self.param_sigma_var, self.param_epsilon_var),
               'ode': self.f_sym, 'quad': self.c_sym}
        opts = {'t0': 0., 'tf': self.dt}
        self.I_fnc = ca.integrator('I', 'cvodes', dae, opts)

    def eval_model_derivs(self):
        def ode_state_sensitivity(sym_args_path_list):
            state_var, action_var, p_mu_var, p_sigma_var, p_eps_var = sym_args_path_list

            ode_p_var = ca.vertcat(action_var, p_mu_var, p_sigma_var, p_eps_var)

            # Jacobian: Adjoint sensitivity
            I_adj = self.I_fnc.factory('I_sym_adj', ['x0', 'p', 'adj:xf', 'adj:qf'], ['adj:x0', 'adj:p'])
            res_sens_xf = I_adj(x0=state_var, p=ode_p_var, adj_xf=np.eye(self.s_dim), adj_qf=0)
            dxfdx = res_sens_xf['adj_x0'].T
            dxfdp = res_sens_xf['adj_p'].T
            dxfdu, dxfdpm, dxfdps, dxfdpe = ca.horzsplit(dxfdp, np.cumsum([0, self.a_dim, self.p_dim, self.p_dim, self.p_dim]))

            # dx = fdt + Fc dw
            Fc = dxfdpe / np.sqrt(self.dt) # SDE correction

            # Taking Jacobian w.r.t ode sensitivity is computationally heavy
            # Instead, compute Hessian of system (d2f/dpedx, d2f/dpedu)
            Fc_direct = ca.jacobian(self.f_sym, self.param_epsilon_var) * np.sqrt(self.dt)

            dFcdx = [ca.jacobian(Fc_direct[:, i], state_var) for i in range(self.p_dim)]
            dFcdu = [ca.jacobian(Fc_direct[:, i], action_var) for i in range(self.p_dim)]

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
        f_derivs = ca.Function('f_derivs', self.path_sym_args, [self.f_sym, dxfdx, dxfdu])  # ["F", "Fx", "Fu", "Fxx", "Fxu", "Fuu"]
        Fc_derivs = ca.Function('Fc_derivs', self.path_sym_args, [Fc, *dFcdx, *dFcdu])
        c_derivs = ca.Function('c_derivs', self.path_sym_args, [self.c_sym] + ode_cost_sensitivity(self.path_sym_args))  #["L", "Lx", "Lu", "Lxx", "Lxu", "Luu"]

        """g, cT: computed from pointwise differentiation"""
        cT_derivs = ca.Function('cT_derivs', self.term_sym_args, [self.cT_sym] + jac_hess_eval(self.cT_sym, self.state_var, None))  # ["LT", "LTx", "LTxx"]

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

    def plot_trajectory(self, traj_data_history, plot_episode, controller_name, save_path):
        variable_tag = [r'$C_{A}[mol/L]$', r'$C_{B}[mol/L]$', r'$T_{R}[^\circ C]$', r'$T_{C}[^\circ C]$',
                        r'$\dot{V}/V_{R}[h^{-1}]$', r'$\dot{Q}[kJ/h]$',
                        r'$\Delta\dot{V}/V_{R}[h^{-1}]$', r'$\Delta\dot{Q}[kJ/h]$']
        time = traj_data_history[0, :, 0] * 60  # minute
        ref = traj_data_history[0, :, -1]
        num_saved_epi = traj_data_history.shape[0]

        fig1, ax1 = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
        fig1.subplots_adjust(hspace=.4, wspace=.5)
        ax1.flat[1].plot(time, ref, 'r--', label='Set point')
        for i in range(self.s_dim + self.a_dim - 1):
            ax1.flat[i].set_xlabel(r'time[$min$]')
            ax1.flat[i].set_ylabel(variable_tag[i])
            for j in range(num_saved_epi):
                epi = plot_episode[j]
                ax1.flat[i].plot(time, traj_data_history[j, :, i+1], label=f'Episode {epi}')
            ax1.flat[i].legend()
            ax1.flat[i].grid()
        fig1.tight_layout()
        plt.savefig(os.path.join(save_path, f'{self.env_name}_{controller_name}_var_traj.png'))
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(time, ref, 'r--', label='Set point')
        ax2.set_xlabel(r'time[$min$]')
        ax2.set_ylabel(variable_tag[1])
        for j in range(num_saved_epi):
            epi = plot_episode[j]
            ax2.plot(time, traj_data_history[j, :, 2], label=f'Episode {epi}')
        ax2.legend()
        ax2.grid()
        fig2.tight_layout()
        plt.savefig(os.path.join(save_path, f'{self.env_name}_{controller_name}_CV_traj.png'))
        plt.show()
