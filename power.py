import torch
from torch.distributions import Normal
import numpy as np
import scipy as sp

from replay_buffer import ReplayBuffer


class PoWER(object):
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
        self.batch_epi = self.config.hyperparameters['batch_epi']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT, batch_size=self.nT)

        # Actor network
        self.actor_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type)
        self.theta = torch.randn([self.rbf_dim, self.a_dim])
        self.sigma = torch.ones([self.rbf_dim, self.a_dim])

        self.phi_traj = np.zeros([self.batch_epi, self.nT, self.rbf_dim])
        self.q_traj = np.inf * np.ones([self.batch_epi, self.nT])
        self.param = torch.zeros([self.batch_epi, self.rbf_dim, self.a_dim])
        self.temp_epsilon_traj = torch.zeros([self.nT, self.rbf_dim, self.a_dim])

    def ctrl(self, epi, step, s, a):
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.ctrl(epi, step, s, a)
            a_val = self.explorer.sample(epi, step, a_nom)
        else:
            a_val = self._choose_action(s, step)

        a_val = np.clip(a_val, -1., 1.)

        return a_val

    def _choose_action(self, s, step):
        # numpy to torch
        s = torch.from_numpy(s.T).float()

        phi = self.actor_net(s)
        epsilon = self.temp_epsilon_traj[step]
        a = phi @ (self.theta + epsilon)

        # torch to numpy
        a = a.T.cpu().detach().numpy()

        return a

    def add_experience(self, *single_expr):
        pass

    def sampling(self, epi):
        epsilon_distribution = Normal(torch.zeros([self.rbf_dim, self.a_dim]), self.sigma)
        self.temp_epsilon_traj = epsilon_distribution.sample([self.nT])

        t, s, _, a = self.env.reset()
        for i in range(self.nT):
            a = self.ctrl(epi, i, s, a)
            t2, s2, _, r, is_term, _ = self.env.step(t, s, a)
            self.replay_buffer.add(*[s, a, r, s2, is_term])
            t, s = t2, s2

    def train(self):
        s_traj, _, r_traj, _, _ = self.replay_buffer.sample_sequence()
        self._estimate(s_traj, r_traj)

    def _estimate(self, s_traj, r_traj):
        # Unbiased estimate of Q function
        r_traj = r_traj.detach().numpy()
        q_traj = [r_traj[-1]]
        for t in range(self.nT):
            q_traj.append(r_traj[-t-1] + q_traj[-1])
        q_traj.reverse()
        q_traj = np.array(q_traj[:-1]).reshape(-1, )

        # Values of basis functions
        phi_traj = self.actor_net(s_traj).detach().numpy()

        # Add high-importance roll-outs information
        if np.any(q_traj[0] < self.q_traj[:, 0]):
            idx = np.argmax(self.q_traj[:, 0])  # argmax if r is cost, argmin if r is reward
            self.q_traj[idx] = q_traj
            self.phi_traj[idx] = phi_traj
            self.param[idx] = self.theta

        self.replay_buffer.clear()

    def _reweight(self, s_traj):
        # Compute importance weights and reweight rollouts
        w_traj = []
        phi_traj = self.actor_net(s_traj).detach().numpy()
        for t in range(self.nT):
            phi = phi_traj[t, :].reshape(-1, 1)
            w = phi @ phi.T / (phi.T @ phi)
            w_traj.append(w)
        w_traj = np.array(w_traj)

        return w_traj

    def _update_actor(self, q_traj, w_traj):
        # Update the policy parameters (theta)
        theta_denom = np.ones((self.rbf_dim, self.rbf_dim))
        theta_num = np.ones((self.rbf_dim, self.a_dim))
        for t in range(self.nT):
            w = w_traj[t, :, :].squeeze()
            q = q_traj[t, :].item()
            epsilon = self.epsilon_traj[t, :, :].squeeze().detach().numpy()

            theta_denom += w * q
            theta_num += w @ epsilon * q

        del_theta = np.linalg.inv(theta_denom) @ theta_num
        self.theta += del_theta

        # Update the standard deviation (sigma)

