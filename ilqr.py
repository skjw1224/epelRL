import scipy as sp
import numpy as np


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

        self.dx_derivs = self.env.dx_derivs
        self.c_derivs = self.env.c_derivs
        self.cT_derivs = self.env.cT_derivs

        self.p_mu, self.p_sigma, self.p_eps = self.env.p_mu, self.env.p_sigma, self.env.p_eps

        # Hyperparameters
        self.learning_rate = self.config.hyperparameters['learning_rate']

        # Trajectory info: x, u, Fx, Fu
        self.traj_derivs_old = None
        self.traj_derivs_new = None

    def ctrl(self, epi, step, x, u):
        if self.traj_derivs_new is None: # Initial control
            self.initial_traj(x, u)

        if step == 0:
            self.backward_sweep()
        xd, ud, _, _ = self.traj_derivs_old[step]
        l, Kx = self.gains[step]

        # Feedback
        delx = x - xd
        u_val = np.clip(ud + (self.learning_rate * l + Kx @ delx), -1, 1)
        return u_val

    def initial_traj(self, x, u):
        x0, u0 = x, u
        _, Fx0, Fu0 = [_.full() for _ in self.dx_derivs(x0, u0, self.p_mu, self.p_sigma, self.p_eps)]
        # Initial control: Assume x0, u0 for whole trajectory
        self.traj_derivs_new = [[x0, u0, Fx0, Fu0] for _ in range(self.nT + 1)]

    def add_experience(self, *single_expr):
        x, u, r, x2, is_term = single_expr
        _, Fx, Fu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
        self.traj_derivs_new.append((x, u, Fx, Fu))

    def backward_sweep(self):
        # Riccati equation solving
        xT, uT, FxT, FuT = self.traj_derivs_new[-1]
        _, LTx, LTxx = [_.full() for _ in self.cT_derivs(xT, self.p_mu, self.p_sigma, self.p_eps)]

        Vxx = LTxx
        Vx = LTx
        V = np.zeros([1, 1])
        self.gains = []
        self.traj_derivs_old = [[np.copy(self.traj_derivs_new[-1][j]) for j in range(len(self.traj_derivs_new[-1]))]]
        for i in reversed(range(self.nT)): # Backward sweep
            x, u, Fx, Fu = self.traj_derivs_new[i]
            self.traj_derivs_old.append([np.copy(self.traj_derivs_new[i][j])
                                         for j in range(len(self.traj_derivs_new[i]))])

            L, Lx, Lu, Lxx, Lxu, Luu = [_.full() for _ in self.c_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]

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

            V = Q + l.T @ Qu + 0.5 * l.T @ Quu @ l
            Vx = Qx + Qxu @ l + Kx.T @ Quu @ l + Kx.T @ Qu
            Vxx = Qxx + Qxu @ Kx + Kx.T @ Quu @ Kx + Kx.T @ Qxu.T

        # Backward seep finish: Reverse gain list
        self.gains.reverse()
        self.traj_derivs_old.reverse()
        self.traj_derivs_new = [[np.copy(self.traj_derivs_new[0][j]) for j in range(len(self.traj_derivs_new[0]))]]

    def train(self, step):
        if hasattr(self, 'gains'):
            l, _ = self.gains[step]
            l = l[0,0]
        else:
            l = 0.
        loss = l
        return loss