import torch
from torch.distributions import Normal
import numpy as np

from replay_buffer.replay_buffer import ReplayBuffer


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
        self.variance_update = self.config.hyperparameters['variance_update']

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
        self.var = np.ones([self.batch_epi, self.rbf_dim, self.a_dim])
        self.temp_epsilon_traj = torch.zeros([self.nT, self.rbf_dim, self.a_dim])
        self.epsilon_traj = torch.zeros([self.batch_epi, self.nT, self.rbf_dim, self.a_dim])

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

    def sampling(self, epi):
        epsilon_distribution = Normal(torch.zeros([self.rbf_dim, self.a_dim]), self.sigma)
        self.temp_epsilon_traj = epsilon_distribution.sample([self.nT])

        t, s, _, a = self.env.reset()
        for i in range(self.nT):
            a = self.ctrl(epi, i, s, a)
            t2, s2, _, r, is_term, _ = self.env.step(t, s, a)
            self.replay_buffer.add(*[s, a, r, s2, is_term])
            t, s = t2, s2

    def train(self, epi):
        s_traj, _, r_traj, _, _ = self.replay_buffer.sample_sequence()
        q_traj, phi_traj = self._estimate(s_traj, r_traj)
        self._add_high_importance_roll_outs(q_traj, phi_traj)
        theta_num, theta_denom, var_num, var_denom = self._reweight(epi)
        self._update_theta(theta_num, theta_denom)
        if self.variance_update:
            self._update_sigma(var_num, var_denom)

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

        self.replay_buffer.clear()

        return q_traj, phi_traj

    def _add_high_importance_roll_outs(self, q_traj, phi_traj):
        # Add high-importance roll-outs information
        if np.any(q_traj[0] < self.q_traj[:, 0]):
            idx = np.argmax(self.q_traj[:, 0])  # argmax if r is cost, argmin if r is reward
            self.q_traj[idx] = q_traj
            self.phi_traj[idx] = phi_traj
            self.var[idx] = self.sigma.detach().numpy()
            self.epsilon_traj[idx] = self.temp_epsilon_traj

    def _reweight(self, epi):
        # Compute importance weights and reweight rollouts
        theta_num = np.zeros([self.rbf_dim, self.a_dim])
        theta_denom = np.zeros([self.rbf_dim, self.a_dim])

        var_num = np.zeros([self.rbf_dim, self.a_dim])
        var_denom = 0

        for idx in range(min(epi+1, self.batch_epi)):
            phi = self.phi_traj[idx]

            for a in range(self.a_dim):
                sigma = np.broadcast_to(self.var[idx, :, a], phi.shape)
                w_denom = np.sum(phi ** 2 * sigma, axis=1)
                w = phi ** 2 / np.broadcast_to(w_denom.reshape(-1, 1), phi.shape)
                epsilon = self.epsilon_traj[idx, :, :, a].detach().numpy()
                q = np.broadcast_to(self.q_traj[idx].reshape(-1, 1), phi.shape)

                theta_num[:, a] += np.sum(w * epsilon * q, axis=0)
                theta_denom[:, a] += np.sum(w * q, axis=0)
                var_num[:, a] += np.sum(epsilon ** 2 * q, axis=0)
            var_denom += np.sum(self.q_traj[idx])

        return theta_num, theta_denom, var_num, var_denom

    def _update_theta(self, theta_num, theta_denom):
        # Update the policy parameters (theta)
        del_theta = theta_num / (theta_denom + 1e-10)
        del_theta = torch.from_numpy(del_theta).float()
        self.theta += del_theta

    def _update_sigma(self, var_num, var_denom):
        # Update the standard deviation (sigma)
        sigma = var_num / (var_denom + 1e-10)
        sigma = np.maximum(sigma, 0.1*np.ones([self.rbf_dim, self.a_dim]))
        sigma = np.minimum(sigma, 10*np.ones([self.rbf_dim, self.a_dim]))
        self.sigma = torch.from_numpy(sigma).float()
