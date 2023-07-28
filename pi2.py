import numpy as np
import torch
from torch.distributions import Normal

from replay_buffer import ReplayBuffer


class PI2(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = 'cpu'

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        # Hyperparameters
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.rbf_dim = self.config.hyperparameters['rbf_dim']
        self.rbf_type = self.config.hyperparameters['rbf_type']
        self.num_rollout = self.config.hyperparameters['num_rollout']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)

        # Actor network
        self.actor_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type)
        self.theta = torch.randn([self.rbf_dim, self.a_dim])
        self.sigma = torch.ones([self.rbf_dim, self.a_dim])

        self.cost_traj = np.zeros([self.num_rollout, self.nT])
        self.epsilon_traj = torch.zeros([self.num_rollout, self.nT, self.rbf_dim, self.a_dim])
        self.g_traj = torch.zeros([self.num_rollout, self.nT, self.rbf_dim])

    def sample(self):
        mean = torch.zeros([self.rbf_dim, self.a_dim])
        epsilon_distribution = Normal(mean, self.sigma)
        self.epsilon_traj = epsilon_distribution.sample([self.num_rollout, self.nT])

        for k in range(self.num_rollout):
            t, s, _, _ = self.env.reset()
            for i in range(self.nT):
                a = self._sample_action(k, i, s)
                t2, s2, _, r, is_term, _ = self.env.step(t, s, a)
                self.cost_traj[k, i] = r
                t, s = t2, s2

    def _sample_action(self, k, i, s):
        # numpy to torch
        s = torch.from_numpy(s.T).float()

        g = self.actor_net(s)
        self.g_traj[k, i] = g
        epsilon = self.epsilon_traj[k, i]
        a = g @ (self.theta + epsilon)

        # torch to numpy
        a = a.T.cpu().detach().numpy()

        return a

    def train(self):
        s_traj, m_traj = self._compute_path_cost()
        p_traj = self._compute_probability(s_traj)
        self._update_parameter(m_traj, p_traj)

    def _compute_path_cost(self):
        s_traj = np.zeros([self.num_rollout, self.nT - 1])
        m_traj = np.zeros([self.num_rollout, self.nT - 1, self.rbf_dim, self.rbf_dim])

        for k in range(self.num_rollout):
            step_cost = self.cost_traj[k, -1]
            stochastic_cost = 0.
            for i in range(self.nT-2, -1, -1):
                step_cost += self.cost_traj[k, i]

                g = self.g_traj[k, i].T.cpu().detach().numpy()
                M = (np.inv(self.R) @ g @ g.T) / (g.T @ np.inv(self.R) @ g)
                m_traj[k, i] = M
                epsilon = self.epsilon_traj[k, i].cpu().detach().numpy()
                stochastic_cost += 0.5 * (self.theta + M @ epsilon).T @ self.R @ (self.theta + M @ epsilon)

                s_traj[k, i] = step_cost + stochastic_cost

        return s_traj, m_traj

    def _compute_probability(self, s_traj):
        s_max = np.max(s_traj)
        s_min = np.min(s_traj)
        s_exp_traj = np.exp(-self.h * (s_traj - s_min) / (s_max - s_min))
        p_traj = s_exp_traj / np.sum(s_exp_traj, axis=0)

        return p_traj

    def _update_parameter(self, m_traj, p_traj):
        del_theta_lst = []
        weight_lst = []

        for i in range(self.nT-1):
            del_theta = 0.
            for k in range(self.num_rollout):
                prob = p_traj[k, i]
                M = m_traj[k, i]
                eps = self.epsilon_traj[k, i]
                del_theta += prob * (M @ eps)
            weight = self.nT - i

            del_theta_lst.append(weight * del_theta)
            weight_lst.append(weight)

        del_theta = torch.from_numpy(np.sum(del_theta_lst) / np.sum(weight_lst)).float()
        self.theta += del_theta

    def evaluate(self):
        pass

