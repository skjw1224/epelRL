import abc
import casadi as ca
import numpy as np
from functools import partial


class Environment(object, metaclass=abc.ABCMeta):
    """
    Base class for environment
    """
    def __init__(self, config):
        self.s_dim = None
        self.a_dim = None
        self.o_dim = None
        self.p_dim = None

        self.dt = None
        self.nT = None

        self.xmin = None
        self.xmax = None
        self.umin = None
        self.umax = None
        self.ymin = None
        self.ymax = None

    @abc.abstractmethod
    def reset(self):
        """
        Reset environment into initial state
        """
        pass

    @abc.abstractmethod
    def step(self, state, action):
        """
        Compute next state from current state and action
        """
        pass

    @abc.abstractmethod
    def system_functions(self):
        """
        Equations that describe dynamics of the system
        """
        pass

    @abc.abstractmethod
    def cost_functions(self, data_type):
        """
        Compute the cost (reward) of current state and action
        """
        pass

    def _set_sym_expressions(self):
        """Syms: Symbolic expressions, Fncs: Symbolic input/output structures"""
        # MX variables for dae function object (no SX)
        self.state_var = ca.MX.sym('x', self.s_dim)
        self.action_var = ca.MX.sym('u', self.a_dim)
        self.param_mu_var = ca.MX.sym('p_mu', self.p_dim)
        self.param_sigma_var = ca.MX.sym('p_sig', self.p_dim)
        self.param_epsilon_var = ca.MX.sym('p_eps', self.p_dim)

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
        dae = {
            'x': self.state_var,
            'p': ca.vertcat(self.action_var, self.param_mu_var, self.param_sigma_var, self.param_epsilon_var),
            'ode': self.f_sym,
            'quad': self.c_sym
        }
        opts = {'t0': 0., 'tf': self.dt}
        self.I_fnc = ca.integrator('I', 'cvodes', dae, opts)

    def _eval_model_derivs(self):
        """f, c: computed from ode sensitivity"""
        dxfdx, dxfdu, dxfdpm, dxfdps, Fc, dFcdx, dFcdu = self._ode_state_sensitivity()
        self.dx_derivs = ca.Function('f_derivs', self.path_sym_args, [self.f_sym, dxfdx, dxfdu])  # ["F", "Fx", "Fu", "Fxx", "Fxu", "Fuu"]
        self.Fc_derivs = ca.Function('Fc_derivs', self.path_sym_args, [Fc, *dFcdx, *dFcdu])
        self.c_derivs = ca.Function('c_derivs', self.path_sym_args, [self.c_sym] + self._ode_cost_sensitivity())  #["L", "Lx", "Lu", "Lxx", "Lxu", "Luu"]

        """g, cT: computed from pointwise differentiation"""
        self.cT_derivs = ca.Function('cT_derivs', self.term_sym_args, [self.cT_sym] + self._jac_hess_eval(self.cT_sym, self.state_var, None))  # ["LT", "LTx", "LTxx"]

    def _ode_state_sensitivity(self):
        ode_p_var = ca.vertcat(self.action_var, self.param_mu_var, self.param_sigma_var, self.param_epsilon_var)

        # Jacobian: Adjoint sensitivity
        I_adj = self.I_fnc.factory('I_sym_adj', ['x0', 'p', 'adj:xf', 'adj:qf'], ['adj:x0', 'adj:p'])
        res_sens_xf = I_adj(x0=self.state_var, p=ode_p_var, adj_xf=np.eye(self.s_dim), adj_qf=0)
        dxfdx = res_sens_xf['adj_x0'].T
        dxfdp = res_sens_xf['adj_p'].T
        dxfdu, dxfdpm, dxfdps, dxfdpe = ca.horzsplit(dxfdp, np.cumsum([0, self.a_dim, self.p_dim, self.p_dim, self.p_dim]))

        # dx = fdt + Fc dw
        Fc = dxfdpe / np.sqrt(self.dt) # SDE correction

        # Taking Jacobian w.r.t ode sensitivity is computationally heavy
        # Instead, compute Hessian of system (d2f/dpedx, d2f/dpedu)
        Fc_direct = ca.jacobian(self.f_sym, self.param_epsilon_var) * np.sqrt(self.dt)

        dFcdx = [ca.jacobian(Fc_direct[:, i], self.state_var) for i in range(self.p_dim)]
        dFcdu = [ca.jacobian(Fc_direct[:, i], self.action_var) for i in range(self.p_dim)]

        return dxfdx, dxfdu, dxfdpm, dxfdps, Fc, dFcdx, dFcdu

    def _ode_cost_sensitivity(self):
        ode_p_var = ca.vertcat(self.action_var, self.param_mu_var, self.param_sigma_var, self.param_epsilon_var)

        # Jacobian: Adjoint sensitivity
        I_adj = self.I_fnc.factory('I_sym_adj', ['x0', 'p', 'adj:xf', 'adj:qf'], ['adj:x0', 'adj:p'])
        res_sens_qf = I_adj(x0=self.state_var, p=ode_p_var, adj_xf=0, adj_qf=1)
        dcdx = res_sens_qf['adj_x0']
        dcdu = res_sens_qf['adj_p'][:self.a_dim]

        d2cdx2 = ca.jacobian(dcdx, self.state_var)
        d2cdxu = ca.jacobian(dcdx, self.action_var)
        d2cdu2 = ca.jacobian(dcdu, self.action_var)

        return [dcdx, dcdu, d2cdx2, d2cdxu, d2cdu2]

    def _jac_hess_eval(self, fnc, x_var, u_var):
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

    def scale(self, var, min, max, shift=True):
        if self.zero_center_scale == True:  # [min, max] --> [-1, 1]
            shifting_factor = max + min if shift else 0.
            scaled_var = (2. * var - shifting_factor) / (max - min)
        else:  # [min, max] --> [0, 1]
            shifting_factor = min if shift else 0.
            scaled_var = (var - shifting_factor) / (max - min)

        return scaled_var

    def descale(self, scaled_var, min, max):
        if self.zero_center_scale == True:  # [-1, 1] --> [min, max]
            var = (max - min) / 2 * scaled_var + (max + min) / 2
        else:  # [0, 1] --> [min, max]
            var = (max - min) * scaled_var + min

        return var
