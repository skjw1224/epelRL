import torch
from functools import partial
from torchdiffeq import odeint  # source: github.com/rtqichen/torchdiffeq
import numpy as np
import utils
import sympy as sym


class CstrEnv(object):
    def __init__(self, device):
        self.device = device

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

        self.param_real = torch.tensor([[self.k10, self.k20, self.k30, self.delHRab, self.delHRbc, self.delHRad]], device=device)
        _, self.param_num = self.param_real.shape

        self.k10_mu_prior = 1.327e+12
        self.k10_sigma_prior = np.log(0.12e+12)
        self.k20_mu_prior = 1.247e+12
        self.k20_sigma_prior = np.log(0.12e+12)
        self.k30_mu_prior = 8.773e+9
        self.k30_sigma_prior = np.log(0.81e+9)
        self.delHRab_mu_prior = 1.84
        self.delHRab_sigma_prior = np.log(4.72)
        self.delHRbc_mu_prior = -9.09
        self.delHRbc_sigma_prior = 3.84
        self.delHRad_mu_prior = -43.26
        self.delHRad_sigma_prior = 4.23

        self.param_mu_prior = torch.tensor(
            [[self.k10_mu_prior, self.k20_mu_prior, self.k30_mu_prior, self.delHRab_mu_prior, self.delHRbc_mu_prior,
              self.delHRad_mu_prior]], device=device)
        self.param_sigma_prior = torch.tensor(
            [[self.k10_sigma_prior, self.k20_sigma_prior, self.k30_sigma_prior, self.delHRab_sigma_prior,
              self.delHRbc_sigma_prior, self.delHRad_sigma_prior]], device=device)

        self.s_dim = 7
        self.a_dim = 2
        self.o_dim = 1
        self.p_dim = self.param_num
        self.dims = (self.s_dim, self.a_dim, self.o_dim, self.p_dim)

        self.real_env = False

        self.n_cond=1
        self.x0 = torch.tensor([[0., 2.1404, 1.4, 387.34, 386.06, 14.19, -1113.5]], dtype=torch.float,  requires_grad=True, device=device)
        # self.x0=self.x0*torch.normal(mean=env.x0.detach()*0+1,std=0.1*torch.tensor([[0., 1., 1., 1., 1., 1., 1.]], dtype=torch.float))
        self.u0 = torch.tensor([[0., 0.]], dtype=torch.float, requires_grad=True, device=device)
        self.t0 = 0.
        self.dt = 20 / 3600.  # hour
        self.tT = 3600 / 3600.  # terminal time
        self.nT = int(self.tT / self.dt) + 1  # episode length

        self.xmin = torch.tensor([[self.t0, 0.001, 0.001, 353.15, 363.15, 3., -9000.]], dtype=torch.float, device=device)
        self.xmax = torch.tensor([[self.tT, 3.5, 1.8, 413.15, 408.15, 35., 0.]], dtype=torch.float, device=device)
        self.umin = torch.tensor([[-1., -1000.]], dtype=torch.float, device=device) / self.dt
        self.umax = torch.tensor([[1., 1000.]], dtype=torch.float, device=device) / self.dt

        self.ymin = torch.tensor([[self.xmin[0, 2]]], dtype=torch.float, device=device)
        self.ymax = torch.tensor([[self.xmax[0, 2]]], dtype=torch.float, device=device)

        Q_elements = torch.tensor([5.], device=device)
        R_elements = torch.tensor([0.1, 0.1], device=device)
        H_elements = torch.tensor([0.], device=device)
        self.Q = torch.diag(Q_elements)
        self.R = torch.diag(R_elements)
        self.H = torch.diag(H_elements)

        "Functions: dx, y, c, cT, dfdx, dfdu, dcdx, dcdu, d2cdx2, d2cdxu, d2cdu2, dcTdx, d2cTdx2"
        self.dx_eval, self.y_eval, self.c_eval, self.cT_eval, self.dx_derivs, self.c_derivs, self.cT_derivs = self.model_derivs()
        self.dx_eval_sym, self.y_eval_sym, self.c_eval_sym, self.cT_eval_sym, self.dx_derivs_sym, self.c_derivs_sym, self.cT_derivs_sym = self.model_derivs_sym()

    def reset(self):
        state = utils.scale(self.x0, self.xmin, self.xmax)
        action = utils.scale(self.u0, self.umin, self.umax)
        obsv = self.y_eval(state, action)
        data_type = 'path'

        return state, obsv, action, data_type

    def ref_traj(self):
        # ref = 0.145*np.cos(2*np.pi*t) + 0.945 # Cos func btw 1.09 ~ 0.8
        # return np.reshape(ref, [1, -1])
        return torch.tensor([0.95], device=self.device)

    def step(self, state, action, *par_args):
        # Real parameter if no parameter arguments are given
        is_realenv = True if len(par_args) == 0 else False

        # Scaled state, action, output
        x = state.detach().requires_grad_()
        u = action.detach().requires_grad_()
        t = utils.descale(x, self.xmin, self.xmax)[0][0].detach()

        # Identify data_type
        if t <= self.tT - 0.5 * self.dt:  # leg_BC assigned & interior time --> 'path'
            data_type = 'path'
        else:
            data_type = 'terminal'

        if is_realenv:
            p_mu, p_sigma, p_eps = self.param_real, torch.zeros([1, self.p_dim], device=self.device), \
                                   torch.zeros([1, self.p_dim], device=self.device)
        else:
            p_mu, p_sigma, p_eps = par_args

        # Integrate ODE
        if data_type == 'path':
            # input: x, u: [1, s] --> odeinput: x: [s, ] --> output: x: [1, s]
            u = torch.clamp(u, -2., 2.)
            f_x = lambda t, x: self.dx_eval(t, x, u, p_mu, p_sigma, p_eps)
            sol_x = odeint(f_x, x, torch.tensor([t, t + self.dt], device=self.device), method='dopri5')
            xplus = sol_x[1]
            xplus = torch.clamp(xplus, -2., 2.)
            costs = self.c_eval(xplus, u) * self.dt

            # Model/Cost derivs
            dfdx, dfdu = self.dx_derivs(x, u, p_mu, p_sigma, p_eps)
            dcdx, _, _, _, d2cdu2 = self.c_derivs(x, u)

            # Derivs in discrete time
            dfdx_DT = torch.eye(self.s_dim, device=self.device) + dfdx * self.dt
            dfdu_DT, dcdx_DT, d2cdu2_DT = dfdu * self.dt, dcdx, d2cdu2
            d2cdu2_DT_inv = torch.cholesky_inverse(torch.cholesky(d2cdu2_DT))
            derivs = [dfdx_DT, dfdu_DT, dcdx_DT, d2cdu2_DT_inv]

            # Terminal?
            is_term = torch.tensor([[False]], dtype=torch.bool, device=self.device)  # Use consistent dimension [1, 1]

        else: # data_type = 'terminal'
            xplus = x
            costs = self.cT_eval(xplus, u)

            # Model/Cost derivs
            dfdx, dfdu = self.dx_derivs(x, u, p_mu, p_sigma, p_eps)
            dcTdx, _ = self.cT_derivs(x)

            # Derivs in discrete time
            dfdx_DT = torch.eye(self.s_dim, device=self.device) + dfdx * self.dt
            dfdu_DT, dcTdx_DT = dfdu * self.dt, dcTdx

            d2cdu2_DT_inv = torch.zeros([self.a_dim, self.a_dim], device=self.device)
            derivs = [dfdx_DT, dfdu_DT, dcTdx_DT, d2cdu2_DT_inv]

            # Terminal?
            is_term = torch.tensor([[True]], dtype=torch.bool, device=self.device) # Use consistent dimension [1, 1]

        yplus = self.y_eval(xplus, u)

        return xplus, yplus, u, costs, is_term, derivs

    def step_model(self, i, state, action, A, B, xd, ud,xd1):
        xd1 = xd1.detach().requires_grad_()
        xd = xd.detach().requires_grad_()
        ud = ud.detach().requires_grad_()
        A = A.detach().requires_grad_()
        B = B.detach().requires_grad_()
        # Scaled state, action, output
        x = state.detach().requires_grad_()
        u = action.detach().requires_grad_()

        t = i*self.dt
        # Identify data_type
        if t <= self.tT - 0.5 * self.dt:  # leg_BC assigned & interior time --> 'path'
            data_type = 'path'
        else:
            data_type = 'terminal'



        # Integrate ODE
        if data_type == 'path':
            # input: x, u: [1, s] --> odeinput: x: [s, ] --> output: x: [1, s]
            u = torch.clamp(u, -2., 2.)
            xdd = torch.matmul(A, (x - xd).T) + torch.matmul(B, (u - ud).T)
            xdd=xdd.T
            xplus=xdd+xd1
            xplus = torch.clamp(xplus, -2., 2.)
            costs = self.c_eval(xplus, u) * self.dt
            # Terminal?
            is_term = torch.tensor([[False]], dtype=torch.bool, device=self.device)  # Use consistent dimension [1, 1]

        else: # data_type = 'terminal'
            xplus = x
            costs = self.cT_eval(xplus, u)
            # Terminal?
            is_term = torch.tensor([[True]], dtype=torch.bool, device=self.device) # Use consistent dimension [1, 1]

        yplus = self.y_eval(xplus, u)

        return xplus, yplus, u, costs, is_term

    def sys_functions(self, t, *args):

        x, u, p_mu, p_sigma, p_eps = args

        k10, k20, k30, E1, E2, E3 = self.k10, self.k20, self.k30, self.E1, self.E2, self.E3
        CA0, T0 = self.CA0, self.T0
        rho, Cp, kw, AR, VR = self.rho, self.Cp, self.kw, self.AR, self.VR
        mk, CpK = self.mk, self.CpK

        x = utils.descale(x, self.xmin, self.xmax)
        u = utils.descale(u, self.umin, self.umax)

        # if the variables become 2D array, then use torch.mm()
        p = p_mu + p_eps * p_sigma
        t, CA, CB, T, TK, VdotVR, QKdot = torch.unbind(x.T)
        dVdotVR, dQKdot = torch.unbind(u.T)

        k10, k20, k30, delHRab, delHRbc, delHRad = torch.unbind(p.T)

        k1 = k10 * (E1 / T).exp()
        k2 = k20 * (E2 / T).exp()
        k3 = k30 * (E3 / T).exp()

        dx = [torch.tensor([1.], dtype=torch.float, device=self.device),
              VdotVR * (CA0 - CA) - k1 * CA - k3 * CA ** 2.,
              -VdotVR * CB + k1 * CA - k2 * CB,
              VdotVR * (T0 - T) - (k1 * CA * delHRab + k2 * CB * delHRbc + k3 * CA ** 2. * delHRad) /
              (rho * Cp) + (kw * AR) / (rho * Cp * VR) * (TK - T),
              (QKdot + (kw * AR) * (T - TK)) / (mk * CpK),
              dVdotVR,
              dQKdot]  # shape [6, n_batch]

        dx = torch.stack(dx)
        dx = torch.transpose(dx, 0, 1)
        dx = utils.scale(dx, self.xmin, self.xmax, shift=False)
        return dx

    def output_functions(self, *args):
        x, u = args
        x = utils.descale(x, self.xmin, self.xmax)

        x = torch.reshape(x, [-1, ])
        y = x[2]
        y = torch.reshape(y, [1, -1])

        y = utils.scale(y, self.ymin, self.ymax)
        return y

    def cost_functions(self, data_type, *args):
        x, u = args

        Q = self.Q
        R = self.R
        H = self.H

        y = self.output_functions(x, u)
        ref = utils.scale(self.ref_traj(), self.ymin, self.ymax)

        if data_type == 'path':
            cost = 0.5 * torch.chain_matmul(y - ref, Q, (y - ref).T) + 0.5 * torch.chain_matmul(u, R, u.T)
        else:  # terminal condition
            cost = 0.5 * torch.chain_matmul(y - ref, H, (y - ref).T)

        return cost

    def model_derivs(self):
        f = self.sys_functions
        y = self.output_functions
        c = partial(self.cost_functions, 'path')
        cT = partial(self.cost_functions, 'terminal')

        def jac_eval(fcn, args, fcn_type):
            x, u, *p_args = args
            x_dim, u_dim = x.shape[-1], u.shape[-1]
            xu = x if u_dim == 0 else torch.cat([x, u], -1)

            f_args = [xu[:, :x_dim], xu[:, x_dim:]]
            if fcn_type == 'f':
                t = [utils.descale(x, self.xmin, self.xmax)[0][0].detach()]
                f_args = t + f_args + p_args

            f_eval = fcn(*f_args)
            jacs = utils.jacobian(f_eval, xu)
            dfdx = jacs[:, :x_dim].detach()
            dfdu = jacs[:, x_dim:].detach()

            if fcn_type == 'c' or fcn_type == 'cT':
                hess = utils.jacobian(jacs, xu)
                d2fdx2 = hess[:x_dim, :x_dim].detach()
                d2fdu2 = hess[x_dim:, x_dim:].detach()
                d2fdxu = hess[:x_dim, x_dim:].detach()
                if fcn_type == 'c':
                    return dfdx.detach_(), dfdu.detach_(), d2fdx2.detach_(), d2fdxu.detach_(), d2fdu2.detach_()
                else: # fcn_type == 'cT':
                    return dfdx.detach_(), d2fdx2.detach_()
            else: # fcn_type == 'f':
                return dfdx.detach_(), dfdu.detach_()
        # 1차, 2차 미분값을 계산
        f_derivs = lambda x, u, *p_args: jac_eval(f, [x, u, *p_args], 'f') #dfdx, dfdu
        c_derivs = lambda x, u: jac_eval(c, [x, u], 'c') # dcdx, dcdu, d2cdx2, d2cdxu, d2cdu2
        cT_derivs = lambda x: jac_eval(cT, [x, torch.zeros([0])], 'cT') # dcTdx, d2cTdx2

        return f, y, c, cT, f_derivs, c_derivs, cT_derivs


    def model_derivs_sym(self):
        # sys_function
        k10, k20, k30, E1, E2, E3 = self.k10, self.k20, self.k30, self.E1, self.E2, self.E3
        CA0, T0 = self.CA0, self.T0
        rho, Cp, kw, AR, VR = self.rho, self.Cp, self.kw, self.AR, self.VR
        mk, CpK = self.mk, self.CpK

        # if the variables become 2D array, then use torch.mm()
        p_mu, p_sigma, p_eps = self.param_real, torch.zeros([1, self.p_dim], device=self.device), \
                               torch.zeros([1, self.p_dim], device=self.device)
        p = p_mu + p_eps * p_sigma

        #     t, CA, CB, T, TK, VdotVR, QKdot = torch.unbind(x.T)
        t_sc, CA_sc, CB_sc, T_sc, TK_sc, VdotVR_sc, QKdot_sc = sym.symbols('t, CA, CB, T, TK, VdotVR, QKdot')
        #     dVdotVR, dQKdot = torch.unbind(u.T)
        dVdotVR_sc, dQKdot_sc = sym.symbols('dVdotVR, dQKdot')

        x_sym = sym.Matrix([t_sc, CA_sc, CB_sc, T_sc, TK_sc, VdotVR_sc, QKdot_sc])
        u_sym = sym.Matrix([dVdotVR_sc, dQKdot_sc])

        xmax = sym.Matrix(self.xmax)
        xmin = sym.Matrix(self.xmin)
        umax = sym.Matrix(self.umax)
        umin = sym.Matrix(self.umin)
        dx_i = sym.Matrix(1 / (self.xmax - self.xmin))
        du_i = sym.Matrix(1 / (self.umax - self.umin))

        x_sym_dc = x_sym.multiply_elementwise((xmax - xmin).T / 2) + (xmax + xmin).T / 2
        u_sym_dc = u_sym.multiply_elementwise((umax - umin).T / 2) + (umax + umin).T / 2

        t, CA, CB, T, TK, VdotVR, QKdot = x_sym_dc
        dVdotVR, dQKdot = u_sym_dc

        k10, k20, k30, delHRab, delHRbc, delHRad = torch.unbind(p.T)
        k10 = k10.numpy()[0]
        k20 = k20.numpy()[0]
        k30 = k30.numpy()[0]
        delHRab = delHRab.numpy()[0]
        delHRbc = delHRbc.numpy()[0]
        delHRad = delHRad.numpy()[0]
        k1 = k10 * sym.exp(E1 / T)
        k2 = k20 * sym.exp(E2 / T)
        k3 = k30 * sym.exp(E3 / T)

        dx_dc = sym.Matrix([1,
                            VdotVR * (CA0 - CA) - k1 * CA - k3 * CA ** 2.,
                            -VdotVR * CB + k1 * CA - k2 * CB,
                            VdotVR * (T0 - T) - (k1 * CA * delHRab + k2 * CB * delHRbc + k3 * CA ** 2. * delHRad) /
                            (rho * Cp) + (kw * AR) / (rho * Cp * VR) * (TK - T),
                            (QKdot + (kw * AR) * (T - TK)) / (mk * CpK),
                            dVdotVR,
                            dQKdot])  # shape [6, n_batch]
        dx = 2 * dx_dc.multiply_elementwise(dx_i.T)
        f_sym=dx
        # output_function
        y_sym = x_sym[2]

        # cost
        Q = sym.Matrix(self.Q)
        R = sym.Matrix(self.R)
        H = sym.Matrix(self.H)
        ref = sym.Matrix(utils.scale(self.ref_traj(), self.ymin, self.ymax))
        c_sym = 0.5* (sym.Matrix([y_sym]) - ref) * Q * (sym.Matrix([y_sym]) - ref).T + 0.5* u_sym.T*R*u_sym
        cT_sym = 0.5* (sym.Matrix([y_sym]) - ref) * H * (sym.Matrix([y_sym]) - ref).T

        # f_derivs
        fx_sym = f_sym.jacobian(x_sym)
        fu_sym = f_sym.jacobian(u_sym)

        # c_derivs
        cx_sym = c_sym.jacobian(x_sym)
        cu_sym = c_sym.jacobian(u_sym)
        cxx_sym = cx_sym.jacobian(x_sym)
        cuu_sym = cu_sym.jacobian(u_sym)
        cxu_sym = cx_sym.jacobian(u_sym)
        # cT_derivs
        cTx_sym = cT_sym.jacobian(x_sym)
        cTxx_sym = cTx_sym.jacobian(x_sym)

        f = sym.lambdify([x_sym, u_sym], f_sym, 'numpy')
        y = sym.lambdify([x_sym, u_sym], y_sym, 'numpy')
        c = sym.lambdify([x_sym, u_sym], c_sym, 'numpy')
        cT = sym.lambdify(x_sym, cT_sym, 'numpy')
        f_derivs = sym.lambdify([x_sym, u_sym], [fx_sym, fu_sym], 'numpy')
        c_derivs = sym.lambdify([x_sym, u_sym], [cx_sym, cu_sym,cxx_sym,cxu_sym,cuu_sym], 'numpy')
        cT_derivs = sym.lambdify([x_sym], [cTx_sym, cTxx_sym], 'numpy')
        return f, y, c, cT, f_derivs, c_derivs, cT_derivs

