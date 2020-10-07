# import torch
import scipy as sp
import scipy.linalg

import numpy as np
import utils

import time

class ILQR(object):
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


        self.dx_derivs = env.dx_derivs
        self.c_derivs = env.c_derivs
        self.cT_derivs = env.cT_derivs

        self.p_mu, self.p_sigma, self.p_eps = env.param_real, env.param_sigma_prior, np.zeros([self.p_dim, 1])

        self.traj_derivs = None

    def ctrl(self, epi, step, x, u):
        if step == 0:
            # to update the AB list from the past episode
            if self.traj_derivs is not None:
                self.traj_derivs.reverse()

            # Loop
            self.prev_traj_derivs = self.traj_derivs
            self.traj_derivs = []

            self.train(x, u, self.prev_traj_derivs)


        xd, ud, _, _ = self.prev_traj_derivs[step]
        l, Kx = self.gains[step]

        delx = x - xd
        u = np.clip(ud + (l + Kx @ delx), -1, 1)
        return u

    def add_experience(self, *single_expr):
        x, u, r, x2, is_term = single_expr
        _, Fx, Fu = self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)
        self.traj_derivs.append((x, u, Fx, Fu))

    def train(self):
        if step == 0:
            if self.traj_derivs is None:
                _, Fx0, Fu0 = self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)
                xd = x
                ud = u
            else:
                xd, ud, Fx, Fu = self.traj_derivs[0]

            # Riccati equation solving

            _, LTx, LTxx = self.cT_derivs(xd, self.p_mu, self.p_sigma, self.p_eps)

            Vxx = LTxx
            Vx = LTx
            V = np.zeros([1, 1])
            self.gains = []
            for i in range(self.nT):
                if self.traj_derivs is None:
                    Fx, Fu= Fx0, Fu0
                else:
                    xd, ud, Fx, Fu = self.traj_derivs[i]

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
                self.gains.append((l, Kx))

                V = Q  + l.T @ Qu + 0.5 * l.T @ Quu @ l
                Vx = Qx + Qxu @ l + Kx.T @ Quu @ l + Kx.T @ Qu
                Vxx = Qxx + Qxu @ Kx + Kx.T @ Quu @ Kx + Kx.T @ Qxu.T

            self.gains.reverse()