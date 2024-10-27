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

    def warm_up_train(self):
        sample = self.rollout_buffer.sample(use_tensor=False)
        states = sample['states']
        u = sample['actions']

        g = self.actor.forward(states)

        theta = np.linalg.solve(g.T@g + self.sigma, g.T@u).T
        self.theta = theta

    def train(self):
        sample = self.rollout_buffer.sample(use_tensor=False)
        rewards = sample['rewards']
        rewards = rewards.reshape(self.num_rollout, self.nT)

        # Compute path cost (S)
        S_traj = np.zeros([self.num_rollout, self.nT])
        step_cost = np.zeros((self.num_rollout, ))
        for i in reversed(range(self.nT)):
            step_cost += rewards[:, i]
            for k in range(self.num_rollout):
                theta_k = self.theta_sample[k].reshape([-1, 1])
                step_cost[k] -= theta_k.T @ theta_k / self.init_lambda / self.rbf_dim**2 / self.num_rollout
            S_traj[:, i] = step_cost

        # Compute probability (P)
        S_max = np.max(S_traj)
        S_min = np.min(S_traj)
        S_exp_traj = np.exp(- self.h * (S_traj - S_min) / (S_max - S_min + 1e-8))
        P_traj = S_exp_traj / np.sum(S_exp_traj, axis=0)

        # Update parameter
        theta_traj = np.zeros([self.nT, self.a_dim, self.rbf_dim])
        sigma_traj = np.zeros([self.nT, self.rbf_dim, self.rbf_dim])
        weight_traj = np.zeros([self.nT])

        for i in range(self.nT):
            theta_i = 0.
            sigma_i = 0.
            for k in range(self.num_rollout):
                theta_dev = self.theta_sample[k] - self.theta
                theta_i += P_traj[k, i] * self.theta_sample[k]
                # theta_i += P_traj[k, i] * theta_dev
                sigma_i += P_traj[k, i] * (theta_dev.T @ theta_dev)
            theta_traj[i] = theta_i
            sigma_traj[i] = sigma_i
            weight_traj[i] = self.nT - i

        theta_prev = self.theta

        d_theta = np.sum(weight_traj.reshape([-1,1,1]) * theta_traj, axis=0) / np.sum(weight_traj) / self.num_rollout
        self.theta = d_theta
        # self.theta = self.theta + self.learning_rate * d_theta
        self.sigma = np.sum(weight_traj.reshape([-1,1,1]) * sigma_traj, axis=0) / np.sum(weight_traj) / self.num_rollout
        self.sigma += self.init_lambda * np.eye(self.rbf_dim)
        self.init_lambda *= 0.99

        loss = np.array([np.linalg.norm(d_theta)])
        # loss = np.array([np.linalg.norm(self.theta - theta_prev)])
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
        np.save(os.path.join(path, '_centers.npy'), self.actor.centers)
        np.save(os.path.join(path, '_shape_params.npy'), self.actor.shape_params)
        np.save(os.path.join(path, '_theta.npy'), self.theta)
        np.save(os.path.join(path, '_sigma.npy'), self.sigma)

    def load(self, path, file_name):
        self.actor.centers = np.load(os.path.join(path, '_centers.npy'))
        self.actor.shape_params = np.load(os.path.join(path, '_shape_params.npy'))
        self.theta = np.load(os.path.join(path, '_theta.npy'))
        self.sigma = np.load(os.path.join(path, '_sigma.npy'))
