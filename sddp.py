# import torch
import scipy as sp
import scipy.linalg

import numpy as np
import utils

import time

class SDDP(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.p_dim = self.env.p_dim

        self.t0 = self.env.t0  # ex) 0
        self.tT = self.env.tT  # ex) 2
        self.nT = self.env.nT
        self.dt = self.env.dt  # ex) dt:0.005


        self.dx_derivs = self.env.dx_derivs
        self.Fc_derivs = self.env.Fc_derivs
        self.c_derivs = self.env.c_derivs
        self.cT_derivs = self.env.cT_derivs

        self.p_mu, self.p_sigma, self.p_eps = self.env.param_real, self.env.param_sigma_prior, np.zeros([self.p_dim, 1])

        self.AB_list_new = None

    def ctrl(self, epi, step, x, u):
        if step == 0:
            # to update the AB list from the past episode
            if self.AB_list_new is not None:
                self.AB_list_new.reverse()
            self.AB_list_old = self.AB_list_new

            self.K_list = self.K_Riccati_ref(x, u, self.AB_list_old)
            self.AB_list_new = []

        l_k, L_k, xd, ud = self.K_list[step]

        delx = x - xd
        u = np.clip(ud + (l_k + L_k @ delx), -1, 1)

        _, A, B = self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)
        Fc_derivs = self.Fc_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)
        Fc = Fc_derivs[0]
        Fcx = Fc_derivs[1:1 + self.p_dim]
        Fcu = Fc_derivs[1 + self.p_dim:]

        # A = torch.eye(self.s_dim, device=self.device) + dfdx * self.dt
        # B = dfdu * self.dt
        self.AB_list_new.append((A, B, Fc, Fcx, Fcu, x, u))
        return u

    def add_experience(self, *single_expr):
        # TODO: Why are we passing this?
        pass

    def K_Riccati_ref(self, x, u, AB_list=None):
        if AB_list is None:
            _, Fx0, Fu0 = self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)
            Fc_derivs = self.Fc_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)
            Fc0 = Fc_derivs[0]
            Fcx0 = Fc_derivs[1:1 + self.p_dim]
            Fcu0 = Fc_derivs[1+self.p_dim:]

            # Fx0 = torch.eye(self.s_dim, device=self.device) + dfdx * self.dt
            # Fu0 = dfdu * self.dt
            xd = x
            ud = u
        else:
            Fx, Fu, Fc, Fcx, Fcu, xd, ud = AB_list[0]

        # Riccati equation solving

        _, LTx, LTxx = self.cT_derivs(xd, self.p_mu, self.p_sigma, self.p_eps)

        Vxx = LTxx
        Vx = LTx
        V = np.zeros([1, 1])
        K_list = []
        for n in range(self.nT):
            if AB_list is None:
                Fx, Fu, Fc, Fcx, Fcu = Fx0, Fu0, Fc0, Fcx0, Fcu0
            else:
                Fx, Fu, Fc, Fcx, Fcu, xd, ud = AB_list[n]

            L, Lx, Lu, Lxx, Lxu, Luu = self.c_derivs(xd, ud, self.p_mu, self.p_sigma, self.p_eps)

            # print('q', Lx)
            # print('r', Lu)
            # print('Lxx', Lxx)
            # print('Luu', Luu)

            U = self.dt * sum([Fc[:, i].T @ Vxx @ Fc[:, i] for i in range(self.p_dim)])  # (1*1)
            Ux = self.dt * sum([Fcx[i].T @ Vxx @ Fc[:, i] for i in range(self.p_dim)])  # (S*1)
            Uu = self.dt * sum([Fcu[i].T @ Vxx @ Fc[:, i] for i in range(self.p_dim)])  # (A*1)
            Uxx = self.dt * sum([Fcx[i].T @ Vxx @ Fcx[i] for i in range(self.p_dim)])  # (S*S)
            Uxu = self.dt * sum([Fcx[i].T @ Vxx @ Fcu[i] for i in range(self.p_dim)])  # (S*A)
            Uuu = self.dt * sum([Fcu[i].T @ Vxx @ Fcu[i] for i in range(self.p_dim)])  # (A*A)

            Q = L + V + 0.5 * U
            Qx = Lx + Fx.T @ Vx + Ux
            Qu = Lu + Fu.T @ Vx + Uu
            Qxx = Lxx + Fx.T @ Vxx @ Fx + Uxx
            Qxu = Lxu + Fx.T @ Vxx @ Fu + Uxu
            Quu = Luu + Fu.T @ Vxx @ Fu + Uuu

            try:
                U = sp.linalg.cholesky(Quu)
                Hi = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, np.eye(len(U)), lower=True))
            except np.linalg.LinAlgError:
                Hi = np.linalg.inv(Quu)

            l = np.clip(- Hi @ Qu, -1, 1)
            Kx = - Hi @ Qxu.T
            K_list.append((l, Kx, xd, ud))

            V = Q + l.T @ Qu + 0.5 * l.T @ Quu @ l
            Vx = Qx + Qxu @ l + Kx.T @ Quu @ l + Kx.T @ Qu
            Vxx = Qxx + Qxu @ Kx + Kx.T @ Quu @ Kx + Kx.T @ Qxu.T

        K_list.reverse()
        return K_list

    def train(self, step):
        if hasattr(self, 'K_list'):
            l, _, _, _ = self.K_list[step]
            l = l[0,0]
        else:
            l = 0.
        loss = l
        return loss
