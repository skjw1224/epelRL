import os
import torch
import scipy.linalg
import numpy as np

from .base_algorithm import Algorithm
from utility.buffer import RolloutBuffer


class SDDP(Algorithm):
    def __init__(self, config):
        self.config = config
        self.device = 'cpu'
        self.s_dim = self.config['s_dim']
        self.a_dim = self.config['a_dim']
        self.p_dim = self.config['p_dim']
        self.nT = self.config['nT']
        self.dt = self.config['dt']

        # Hyperparameters
        self.gamma = self.config['sddp_gamma']

        config['buffer_size'] = self.nT
        config['batch_size'] = self.nT
        self.rollout_buffer = RolloutBuffer(config)

        # Policy gains
        self.step = 0
        self.prev_traj = [np.zeros([self.nT, self.s_dim]), np.zeros([self.nT, self.a_dim])]
        self.gains = [(np.ones([self.a_dim, 1]), np.ones([self.a_dim, self.s_dim])) for _ in range(self.nT)]

        self.loss_lst = ['Qu loss']

    def ctrl(self, state):
        prev_state = self.prev_traj[0][self.step].reshape(-1, 1)
        prev_action = self.prev_traj[1][self.step].reshape(-1, 1)
        k, K = self.gains[self.step]
        action = prev_action + self.gamma * (k + K @ (state - prev_state))
        action = np.clip(action, -1, 1)

        self.step = (self.step + 1) % self.nT
        
        return action

    def add_experience(self, experience):
        self.rollout_buffer.add(experience)

    def warm_up_train(self):
        pass

    def train(self):
        # Replay buffer sample sequence
        sample = self.rollout_buffer.sample(use_tensor=False)
        states = sample['states']
        actions = sample['actions']
        dones = sample['dones']
        derivs = sample['derivs']
        f_x, f_u, l_x, l_u, l_xx, l_xu, l_uu, Fc, Fc_x, Fc_u = derivs
        
        self.prev_traj = [states, actions]

        Qu_loss = 0.
        # Riccati equation solving
        for i in reversed(range(self.nT)):
            if dones[i] or i == self.nT - 1:
                V_x = l_x[i]
                V_xx = l_xx[i]
            else:
                # TODO: add second derivative terms
                U_x = self.dt * sum([Fc_x[i][j].T @ V_xx @ Fc[i, :, j].reshape(-1, 1) for j in range(self.p_dim)])
                U_u = self.dt * sum([Fc_u[i][j].T @ V_xx @ Fc[i, :, j].reshape(-1, 1) for j in range(self.p_dim)])
                U_xx = self.dt * sum([Fc_x[i][j].T @ V_xx @ Fc_x[i][j] for j in range(self.p_dim)])
                U_xu = self.dt * sum([Fc_x[i][j].T @ V_xx @ Fc_u[i][j] for j in range(self.p_dim)])
                U_uu = self.dt * sum([Fc_u[i][j].T @ V_xx @ Fc_u[i][j] for j in range(self.p_dim)])

                Q_x = l_x[i] + f_x[i].T @ V_x + U_x
                Q_u = l_u[i] + f_u[i].T @ V_x + U_u
                Q_xx = l_xx[i] + f_x[i].T @ V_xx @ f_x[i] + U_xx
                Q_xu = l_xu[i] + f_x[i].T @ V_xx @ f_u[i] + U_xu
                Q_uu = l_uu[i] + f_u[i].T @ V_xx @ f_u[i] + U_uu

                try:
                    U = scipy.linalg.cholesky(Q_uu)
                    Q_uu_inv = scipy.linalg.solve_triangular(U, scipy.linalg.solve_triangular(U.T, np.eye(len(U)), lower=True))
                except np.linalg.LinAlgError:
                    Q_uu_inv = np.linalg.inv(Q_uu)

                k = np.clip(- Q_uu_inv @ Q_u, -1, 1)
                K = - Q_uu_inv @ Q_xu.T
                self.gains.append((k, K))

                V_x = Q_x + Q_xu @ k + K.T @ Q_uu @ k + K.T @ Q_u
                V_xx = Q_xx + Q_xu @ K + K.T @ Q_uu @ K + K.T @ Q_xu.T

                Qu_loss += np.linalg.norm(Q_u)

        self.gains.reverse()
        loss = np.array([Qu_loss])

        self.rollout_buffer.reset()

        return loss

    def save(self, path, file_name):
        pass

    def load(self, path, file_name):
        pass