# import torch
import scipy as sp
import scipy.linalg

import numpy as np
import utils

import time

class Ilqr(object):
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.p_dim = env.p_dim

        self.t0 = env.t0  # ex) 0
        self.tT = env.tT  # ex) 2
        self.nT = env.nT
        self.dt = env.dt  # ex) dt:0.005


        self.dx_derivs = env.dx_derivs
        self.Fc_derivs = env.Fc_derivs
        self.c_derivs = env.c_derivs
        self.cT_derivs = env.cT_derivs

        self.p_mu, self.p_sigma, self.p_eps = env.param_real, env.param_sigma_prior, np.zeros([self.p_dim, 1])

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
        # A = torch.eye(self.s_dim, device=self.device) + dfdx * self.dt
        # B = dfdu * self.dt
        self.AB_list_new.append((A, B, x, u))
        return u

    def add_experience(self, *single_expr):
        pass

    def K_Riccati_ref(self, x, u, AB_list=None):
        if AB_list is None:
            _, Fx0, Fu0 = self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)

            # Fx0 = torch.eye(self.s_dim, device=self.device) + dfdx * self.dt
            # Fu0 = dfdu * self.dt
            xd = x
            ud = u
        else:
            Fx, Fu, xd, ud = AB_list[0]

        # Riccati equation solving

        _, LTx, LTxx = self.cT_derivs(xd, self.p_mu, self.p_sigma, self.p_eps)

        Vxx = LTxx
        Vx = LTx
        V = np.zeros([1, 1])
        K_list = []
        for n in range(self.nT):
            if AB_list is None:
                Fx, Fu= Fx0, Fu0
            else:
                Fx, Fu, xd, ud = AB_list[n]

            L, Lx, Lu, Lxx, Lxu, Luu = self.c_derivs(xd, ud, self.p_mu, self.p_sigma, self.p_eps)


            start_time = time.time()

            Q = L + V
            Qx = Lx + Fx.T @ Vx
            Qu = Lu + Fu.T @ Vx
            Qxx = Lxx + Fx.T @ Vxx @ Fx
            Qxu = Lxu + Fx.T @ Vxx @ Fu
            Quu = Luu + Fu.T @ Vxx @ Fu

            try:
                U = sp.linalg.cholesky(Quu)
                Hi = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, np.eye(len(U)), lower=True))
            except np.linalg.LinAlgError:
                Hi = np.linalg.inv(Quu)

            l = np.clip(- Hi @ Qu, -1, 1)
            Kx = - Hi @ Qxu.T
            K_list.append((l, Kx, xd, ud))

            V = Q  + l.T @ Qu + 0.5 * l.T @ Quu @ l
            Vx = Qx + Qxu @ l + Kx.T @ Quu @ l + Kx.T @ Qu
            Vxx = Qxx + Qxu @ Kx + Kx.T @ Quu @ Kx + Kx.T @ Qxu.T

        K_list.reverse()
        return K_list