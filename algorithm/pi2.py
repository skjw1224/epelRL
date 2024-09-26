import os
import numpy as np

from .base_algorithm import Algorithm
from network.rbf import RBF
from utility.buffer import RolloutBuffer


class PI2(Algorithm):
    def __init__(self, config):
        self.config = config
        self.device = 'cpu'
        self.s_dim = self.config['s_dim']
        self.a_dim = self.config['a_dim']
        self.nT = self.config['nT']

        # Hyperparameters
        self.rbf_dim = self.config['rbf_dim']
        self.rbf_type = self.config['rbf_type']
        self.num_rollout = self.config['num_rollout']
        self.h = self.config['h']
        self.init_lambda = self.config['init_lambda']

        config['buffer_size'] = self.nT * self.num_rollout
        config['batch_size'] = self.nT * self.num_rollout
        self.rollout_buffer = RolloutBuffer(config)

        # Actor network
        self.actor = RBF(self.s_dim, self.rbf_dim, self.rbf_type)
        self.theta = np.random.randn(self.a_dim, self.rbf_dim)
        self.sigma = self.init_lambda * np.eye(self.rbf_dim)

        self.theta_sample = np.zeros([self.num_rollout, self.a_dim, self.rbf_dim])
        self.cost_traj = np.zeros([self.num_rollout, self.nT])
        self.step = 0
        self.rollout_count = 0

        self.loss_lst = ['Actor loss']

    def ctrl(self, state):
        if self.step == 0:
            for a in range(self.a_dim):
                self.theta_sample[self.rollout_count, a, :] = np.random.multivariate_normal(self.theta[a, :], self.sigma)

        g = self.actor.forward(state.T)  # 1*F
        theta = self.theta_sample[self.rollout_count, :, :]  # A*F
        action = g @ theta.T  # 1*A
        action = np.clip(action.T, -1., 1.)  # A*1

        self.step = (self.step + 1) % self.nT
        if self.step == 0:
            self.rollout_count = (self.rollout_count + 1) % self.num_rollout

        return action
    
    def add_experience(self, experience):
        self.rollout_buffer.add(experience)

    def train(self):
        sample = self.rollout_buffer.sample(use_tensor=False)
        rewards = sample['rewards']
        rewards = rewards.reshape(self.num_rollout, self.nT)

        # Compute path cost (S)
        S_traj = np.zeros([self.num_rollout, self.nT])
        step_cost = np.zeros((self.num_rollout, ))
        for i in reversed(range(self.nT)):
            step_cost += rewards[:, i]
            S_traj[:, i] = step_cost

        # Compute probability (P)
        S_max = np.max(S_traj)
        S_min = np.min(S_traj)
        S_exp_traj = np.exp(- self.h * (S_traj - S_min) / (S_max - S_min))
        P_traj = S_exp_traj / np.sum(S_exp_traj, axis=0)

        # Update parameter
        theta_traj = np.zeros([self.nT, self.a_dim, self.rbf_dim])
        sigma_traj = np.zeros([self.nT, self.rbf_dim, self.rbf_dim])
        weight_traj = np.zeros([self.nT])

        for i in range(self.nT):
            theta_i = 0.
            sigma_i = 0.
            for k in range(self.num_rollout):
                theta_i += P_traj[k, i] * self.theta_sample[k]
                theta_dev = self.theta_sample[k] - self.theta
                sigma_i += P_traj[k, i] * (theta_dev.T @ theta_dev)
            theta_traj[i] = theta_i
            sigma_traj[i] = sigma_i
            weight_traj[i] = self.nT - i

        theta_prev = self.theta

        self.theta = np.sum(theta_traj, axis=0) / np.sum(weight_traj)
        self.sigma = np.sum(sigma_traj, axis=0) / np.sum(weight_traj)
        self.sigma += self.init_lambda * np.eye(self.rbf_dim)

        loss = np.array([np.linalg.norm(self.theta - theta_prev)])
        return loss

    def _compute_path_cost(self):
        s_traj = np.zeros([self.num_rollout, self.nT])
        for k in range(self.num_rollout):
            step_cost = 0.
            for i in range(self.nT-1, -1, -1):
                step_cost += self.cost_traj[k, i]
                s_traj[k, i] = step_cost

        return s_traj

    def save(self, path, file_name):
        pass

    def load(self, path, file_name):
        pass
