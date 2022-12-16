import torch
from functools import partial
from torchdiffeq import odeint  # source: github.com/rtqichen/torchdiffeq
import numpy as np
import utils


class Env(object):
    def __init__(self, device):
        self.device = device

        self.s_dim = 1
        self.a_dim = 1
        self.o_dim = 1
        # self.dims = (self.s_dim, self.a_dim, self.o_dim, self.p_dim)

        self.real_env = False

        self.x0 = torch.tensor([1.0], dtype=torch.float, requires_grad=True, device=device)
        self.u0 = torch.tensor([0.], dtype=torch.float, requires_grad=True, device=device)
        self.t0 = 0.
        self.dt = 10 / 1000.  # hour
        self.tT = 1000 / 1000.  # terminal time
        self.nT = int(self.tT / self.dt) + 1  # episode length

        self.A = torch.tensor([1.0])
        self.B = torch.tensor([1.0])

        self.xmin = torch.tensor([[-100.]], dtype=torch.float, device=device)
        self.xmax = torch.tensor([[100.]], dtype=torch.float, device=device)
        self.umin = torch.tensor([[-100.]], dtype=torch.float, device=device)
        self.umax = torch.tensor([[100.]], dtype=torch.float, device=device)
        # self.umin = torch.tensor([[-100.]], dtype=torch.float, device=device) / self.dt
        # self.umax = torch.tensor([[100.]], dtype=torch.float, device=device) / self.dt
        #
        self.ymin = self.xmin
        self.ymax = self.xmax

        self.Q = torch.tensor([[2.]], device=device)
        self.R = torch.tensor([[2.]], device=device)
        # self.Q = torch.diag(Q_elements)
        # self.R = torch.diag(R_elements)

        "Functions: dx, y, c, cT, dfdx, dfdu, dcdx, dcdu, d2cdx2, d2cdxu, d2cdu2, dcTdx, d2cTdx2"
        self.y_eval, self.c_eval, self.cT_eval = self.model_derivs()

    def reset(self):
        # state = utils.scale(self.x0, self.xmin, self.xmax)
        # action = utils.scale(self.u0, self.umin, self.umax)
        state = torch.tensor([[self.x0]])
        action = self.u0
        obsv = self.y_eval(state, action)
        data_type = 'path'

        return state, obsv, action, data_type

    def ref_traj(self):
        # ref = 0.145*np.cos(2*np.pi*t) + 0.945 # Cos func btw 1.09 ~ 0.8
        # return np.reshape(ref, [1, -1])
        return torch.tensor([0.], device=self.device)

    def step(self, state, action, *par_args):
        # Real parameter if no parameter arguments are given
        is_realenv = True if len(par_args) == 0 else False

        # Scaled state, action, output
        # x = state.detach().requires_grad_()
        # u = action.detach().requires_grad_()
        # t = utils.descale(x, self.xmin, self.xmax)[0][0].detach()
        x = state.detach().requires_grad_()
        u = action.detach().requires_grad_()
        t = np.array([par_args]) * self.dt
        A = self.A
        B = self.B
        derivs = []

        # Identify data_type
        if t <= self.tT - self.dt:  # leg_BC assigned & interior time --> 'path'
            data_type = 'path'
        else:
            data_type = 'terminal'

        # Integrate ODE
        if data_type == 'path':

            xplus = A * x + B * u + 0.1 * torch.randn(1)
            costs = self.c_eval(xplus, u)

            # Terminal?
            is_term = torch.tensor([[False]], dtype=torch.bool, device=self.device)  # Use consistent dimension [1, 1]

        else: # data_type = 'terminal'

            xplus = x
            costs = self.cT_eval(xplus, u)

            # Terminal?
            is_term = torch.tensor([[True]], dtype=torch.bool, device=self.device) # Use consistent dimension [1, 1]

        yplus = self.y_eval(xplus, u)

        return xplus, yplus, u, costs, is_term, derivs

    def output_functions(self, *args):
        x, u = args
        # x = utils.descale(x, self.xmin, self.xmax)

        # x = torch.reshape(x, [-1, ])
        y = x
        # y = torch.reshape(y, [1, -1])

        # y = utils.scale(y, self.ymin, self.ymax)
        return y

    def cost_functions(self, data_type, *args):

        x, u = args
        u = torch.unsqueeze(u, 0)   # A2C, LSTD: on / LQR: off

        Q = self.Q
        R = self.R

        a = 0.5 * torch.chain_matmul(x, Q, x.T) + 0.5 * torch.chain_matmul(u, R, u.T)

        if data_type == 'path':
            cost = 0.5 * torch.chain_matmul(x, Q, x.T) + 0.5 * torch.chain_matmul(u, R, u.T)
        else:  # terminal condition
            cost = 0.5 * torch.chain_matmul(x, Q, x.T)

        return cost

    def model_derivs(self):
        y = self.output_functions
        c = partial(self.cost_functions, 'path')
        cT = partial(self.cost_functions, 'terminal')

        return y, c, cT