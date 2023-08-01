import numpy as np
import torch
from torch.distributions import MultivariateNormal


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
        self.h = self.config.hyperparameters['h']
        self.init_lambda = self.config.hyperparameters['init_lambda']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)

        # Actor network
        self.actor_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type)
        self.theta = torch.randn([self.a_dim, self.rbf_dim])
        self.sigma = self.init_lambda * torch.eye(self.rbf_dim)

        self.theta_lst = torch.zeros([self.num_rollout, self.a_dim, self.rbf_dim])
        self.cost_traj = np.zeros([self.num_rollout, self.nT])

    def sampling(self, epi):
        theta_distribution = MultivariateNormal(self.theta, self.sigma)

        for k in range(self.num_rollout):
            # Sample parameters
            self.theta_lst[k] = theta_distribution.sample()

            # Execute policy
            t, s, _, _ = self.env.reset()
            for i in range(self.nT):
                a = self._sample_action(k, i, s)
                t2, s2, _, r, is_term, _ = self.env.step(t, s, a)
                self.cost_traj[k, i] = r
                t, s = t2, s2

    def _sample_action(self, k, i, s):
        # numpy to torch
        s = torch.from_numpy(s.T).float()

        g = self.actor_net(s)  # 1*F
        theta = self.theta_lst[k, :, :]  # A*F
        a = g @ theta.T  # 1*A

        # torch to numpy
        a = a.T.cpu().detach().numpy()

        return a

    def train(self):
        s_traj = self._compute_path_cost()
        p_traj = self._compute_probability(s_traj)
        self._update_parameter(p_traj)

    def _compute_path_cost(self):
        s_traj = np.zeros([self.num_rollout, self.nT])
        for k in range(self.num_rollout):
            step_cost = 0.
            for i in range(self.nT-1, -1, -1):
                step_cost += self.cost_traj[k, i]
                s_traj[k, i] = step_cost

        return s_traj

    def _compute_probability(self, s_traj):
        s_max = np.max(s_traj)
        s_min = np.min(s_traj)
        s_exp_traj = np.exp(- self.h * (s_traj - s_min) / (s_max - s_min))
        p_traj = s_exp_traj / np.sum(s_exp_traj, axis=0)

        return p_traj

    def _update_parameter(self, p_traj):
        del_theta_lst = []
        sigma_lst = []
        weight_lst = []

        for i in range(self.nT-1):
            del_theta = 0.
            sigma = 0.
            for k in range(self.num_rollout):
                prob = p_traj[k, i]
                M = m_traj[k, i]
                eps = self.epsilon_traj[k, i].cpu().detach().numpy()
                del_theta += prob * (M @ eps)
                sigma += prob * (M @ eps) ** 2
            weight = self.nT - i

            del_theta_lst.append(weight * del_theta)
            sigma_lst.append(weight * sigma)
            weight_lst.append(weight)

        del_theta = torch.tensor(np.sum(del_theta_lst) / np.sum(weight_lst)).float()
        sigma = torch.tensor(np.sum(sigma_lst, axis=0) / np.sum(weight_lst)).float()
        self.theta += del_theta
        self.sigma = sigma

    def evaluate(self):
        pass

