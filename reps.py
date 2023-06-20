import torch
from torch.distributions import Normal
import numpy as np
import scipy.optimize as optim

from replay_buffer import ReplayBuffer


class REPS(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        self.epoch = 0

        # hyperparameters
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.max_kl_divergence = self.config.hyperparameters['max_kl_divergence']
        self.rbf_dim = self.config.hyperparameters['rbf_dim']
        self.rbf_type = self.config.hyperparameters['rbf_type']
        self.batch_epi = self.config.hyperparameters['batch_epi']
        self.num_critic_update = self.config.hyperparameters['num_critic_update']
        self.critic_reg = self.config.hyperparameters['critic_reg']
        self.actor_reg = self.config.hyperparameters['actor_reg']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT*self.batch_epi, batch_size=self.nT*self.batch_epi)

        self.critic_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type).to(self.device)
        self.actor_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type).to(self.device)
        self.actor_log_std = torch.zeros(1, self.a_dim).to(self.device)

        self.eta = torch.rand([1]).to(self.device)
        self.theta = torch.rand([self.rbf_dim, 1]).to(self.device)
        self.omega = torch.rand([self.rbf_dim, self.a_dim]).to(self.device)

    def ctrl(self, epi, step, s, a):
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.ctrl(epi, step, s, a)
            a_val = self.explorer.sample(epi, step, a_nom)
        else:
            a_val = self._choose_action(s)

        a_val = np.clip(a_val, -1., 1.)

        return a_val

    def _choose_action(self, s):
        # numpy to torch
        s = torch.from_numpy(s.T).float().to(self.device)

        mean = torch.matmul(self.actor_net(s), self.omega)
        std = torch.exp(self.actor_log_std)
        a_distribution = Normal(mean, std)
        a = a_distribution.sample()
        a = torch.tanh(a)

        # torch to numpy
        a = a.T.cpu().detach().numpy()

        return a

    def add_experience(self, *single_expr):
        pass

    def sampling(self, epi):
        # Rollout a few episodes for sampling
        for _ in range(self.batch_epi + 1):
            for _ in range(self.nT):
                t, s, _, a = self.env.reset()
                for i in range(self.nT):
                    a = self.ctrl(epi, i, s, a)
                    t2, s2, _, r, is_term, _ = self.env.step(t, s, a)
                    self.replay_buffer.add(*[s, a, r, s2, is_term])
                    t, s = t2, s2

    def train(self, step):
        if step == self.nT - 1:
            # Replay buffer sample
            s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample_sequence()

            # Compute Bellman error and feature difference
            delta, lambd = self._compute_weights(s_batch, r_batch, s2_batch)

            # Compute dual function and the dual function's derivative
            g = self._dual_function
            del_g = self._dual_function_grad

            # Optimize value function
            for _ in range(self.num_critic_update):
                x0 = torch.cat([self.eta, self.theta])
                sol = optim.minimize(g, x0, method='L-BFGS-B', jac=del_g, args=(delta, lambd))
                self.eta, self.theta = sol.x[0], sol.x[1:]

            # Policy update
            self._update_policy(s_batch, a_batch, r_batch, s2_batch)

    def _compute_weights(self, states, rewards, next_states):
        phi_s = self.critic_net(states)
        phi_s2 = self.critic_net(next_states)
        delta = rewards + np.dot(phi_s2 - phi_s, self.theta)
        lambd = np.dot(phi_s2 - phi_s, self.theta)

        return delta, lambd

    def _dual_function(self, var, delta, lambd):
        eta, theta = var[0], var[1:]
        epsilon = self.max_kl_divergence
        g = eta * (epsilon + np.log(np.mean(np.exp(delta / eta))))
        g += self.critic_reg * np.sum(theta ** 2)

        return g

    def _dual_function_grad(self, var, delta, lambd):
        eta, theta = var[0], var[1:]
        epsilon = self.max_kl_divergence
        weights = np.exp(epsilon + delta / eta)
        del_theta = eta * (np.sum(weights * lambd) / np.sum(weights))
        del_eta = np.log(np.sum(weights)) - (1 / eta**2) * (np.sum(weights * delta) / np.sum(weights))
        del_g = np.hstack(del_theta, del_eta)

        return del_g

    def _update_policy(self, states, actions, rewards, next_states):
        delta, _ = self._compute_weights(states, rewards, next_states)
        weights = np.exp(delta / self.eta)
        phi = self.actor_net(states).cpu().detach().numpy() * weights
        self.omega = (phi.T @ phi + self.actor_reg * np.eye(self.rbf_dim)) @ phi.T @ actions
        # TODO: update std
        self.actor_log_std
