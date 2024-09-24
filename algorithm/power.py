import os
import numpy as np

from .base_algorithm import Algorithm
from network.rbf import RBF
from utility.buffer import RolloutBuffer


class PoWER(Algorithm):
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
        self.variance_update = self.config['variance_update']

        config['buffer_size'] = self.nT
        config['batch_size'] = self.nT
        self.rollout_buffer = RolloutBuffer(config)

        # Actor network
        self.actor = RBF(self.s_dim, self.rbf_dim, self.rbf_type)
        self.theta = np.random.randn(self.rbf_dim, self.a_dim)
        self.sigma = np.ones([self.rbf_dim, self.a_dim])

        self.phi_traj = np.zeros([self.num_rollout, self.nT, self.rbf_dim])
        self.q_traj = np.inf * np.ones([self.num_rollout, self.nT])
        self.var = np.ones([self.num_rollout, self.rbf_dim, self.a_dim])
        self.epsilon_traj = np.zeros([self.num_rollout, self.nT, self.rbf_dim, self.a_dim])

        mean = np.zeros((self.nT, self.rbf_dim, self.a_dim))
        std = np.repeat(self.sigma[np.newaxis], self.nT, axis=0)
        self.epsilons = np.random.normal(mean, std)
        self.step = 0
        self.episode = 0

        self.loss_lst = ['Actor loss']

    def ctrl(self, state):
        phi = self.actor.forward(state.T)
        epsilon = self.epsilons[self.step]
        action = phi @ (self.theta + epsilon)
        action = np.clip(action.T, -1., 1.)

        self.step = (self.step + 1) % self.nT

        return action
    
    def add_experience(self, experience):
        self.rollout_buffer.add(experience)

    def train(self):
        sample = self.rollout_buffer.sample(use_tensor=False)
        states = sample['states']
        rewards = sample['rewards']

        # Unbiased estimate of Q function
        q = 0
        q_values = np.zeros(rewards.shape)
        for i, reward in enumerate(reversed(rewards)):
            q += reward
            q_values[-i-1] = q

        # Values of basis functions
        phi = self.actor.forward(states)

        # Add high-importance roll-outs information
        if np.any(q_values[0] < self.q_traj[:, 0]):
            idx = np.argmax(self.q_traj[:, 0])  # argmax if r is cost, argmin if r is reward
            self.q_traj[idx] = q_values.squeeze()
            self.phi_traj[idx] = phi
            self.var[idx] = self.sigma
            self.epsilon_traj[idx] = self.epsilons
        
        # Compute importance weights and reweight rollouts
        theta_num = np.zeros([self.rbf_dim, self.a_dim])
        theta_denom = np.zeros([self.rbf_dim, self.a_dim])

        var_num = np.zeros([self.rbf_dim, self.a_dim])
        var_denom = 0

        self.episode += 1
        for idx in range(min(self.episode, self.num_rollout)):
            phi = self.phi_traj[idx]

            for a in range(self.a_dim):
                sigma = np.broadcast_to(self.var[idx, :, a], phi.shape)
                w_denom = np.sum(phi ** 2 * sigma, axis=1)
                w = phi ** 2 / np.broadcast_to(w_denom.reshape(-1, 1), phi.shape)
                epsilon = self.epsilon_traj[idx, :, :, a]
                q = np.broadcast_to(self.q_traj[idx].reshape(-1, 1), phi.shape)

                theta_num[:, a] += np.sum(w * epsilon * q, axis=0)
                theta_denom[:, a] += np.sum(w * q, axis=0)
                var_num[:, a] += np.sum(epsilon ** 2 * q, axis=0)
            var_denom += np.sum(self.q_traj[idx])

        # Update the policy parameters (theta)
        del_theta = theta_num / (theta_denom + 1e-10)
        self.theta += del_theta

        # Update the standard deviation (sigma)
        if self.variance_update:
            sigma = var_num / (var_denom + 1e-10)
            sigma = np.maximum(sigma, 0.1*np.ones([self.rbf_dim, self.a_dim]))
            sigma = np.minimum(sigma, 10*np.ones([self.rbf_dim, self.a_dim]))
            self.sigma = sigma

        # Update epsilon
        mean = np.zeros((self.nT, self.rbf_dim, self.a_dim))
        std = np.repeat(self.sigma[np.newaxis], self.nT, axis=0)
        self.epsilons = np.random.normal(mean, std)

        self.rollout_buffer.reset()

        loss = np.array([np.linalg.norm(del_theta)])

        return loss

    def save(self, path, file_name):
        pass

    def load(self, path, file_name):
        pass