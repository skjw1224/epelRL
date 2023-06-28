import torch
from torch.distributions import Normal
import numpy as np
import scipy.optimize as optim

from replay_buffer import ReplayBuffer


class REPS(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = 'cpu'

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        # Hyperparameters
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.max_kl_divergence = self.config.hyperparameters['max_kl_divergence']
        self.rbf_dim = self.config.hyperparameters['rbf_dim']
        self.rbf_type = self.config.hyperparameters['rbf_type']
        self.batch_epi = self.config.hyperparameters['batch_epi']
        self.critic_reg = self.config.hyperparameters['critic_reg']
        self.actor_reg = self.config.hyperparameters['actor_reg']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT*self.batch_epi, batch_size=self.nT*self.batch_epi)

        self.critic_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type)
        self.actor_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type)
        self.actor_std = torch.zeros(1, self.a_dim)

        self.eta = torch.rand([1, 1])
        self.theta = torch.rand([self.rbf_dim, 1])
        self.omega = torch.rand([self.rbf_dim, self.a_dim])

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

        phi = self.actor_net(s).type(torch.float64)
        mean = torch.matmul(phi, self.omega)
        a_distribution = Normal(mean, self.actor_std)
        a = a_distribution.sample()
        a = torch.tanh(a)

        # torch to numpy
        a = a.T.detach().numpy()

        return a

    def add_experience(self, *single_expr):
        pass

    def sampling(self, epi):
        # Rollout a few episodes for sampling
        for _ in range(self.batch_epi + 1):
            t, s, _, a = self.env.reset()
            for i in range(self.nT):
                a = self.ctrl(epi, i, s, a)
                t2, s2, _, r, is_term, _ = self.env.step(t, s, a)
                self.replay_buffer.add(*[s, a, r, s2, is_term])
                t, s = t2, s2

    def train(self):
        # Replay buffer sample
        s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample_sequence()

        # Compute Bellman error and feature difference
        delta, lambd = self._compute_weights(s_batch, r_batch, s2_batch)

        # Compute dual function and the dual function's derivative
        g = self._dual_function
        del_g = self._dual_function_grad

        # Optimize value function
        x0 = torch.cat([self.theta, self.eta]).type(torch.float64).cpu()
        sol = optim.minimize(g, x0, method='L-BFGS-B', jac=del_g, args=(delta, lambd),
                             bounds=((1e-16, 1e16),) + ((-np.inf, np.inf),) * self.rbf_dim)
        sol = torch.from_numpy(sol.x).float()
        self.theta, self.eta = sol[:-1].reshape(-1, 1), sol[-1].reshape(-1, 1)

        # Policy update
        self._update_policy(s_batch, a_batch, r_batch, s2_batch)

    def _compute_weights(self, states, rewards, next_states):
        phi_s = self.critic_net(states)
        phi_s2 = self.critic_net(next_states)
        delta = rewards + torch.matmul(phi_s2 - phi_s, self.theta)
        lambd = phi_s2 - phi_s
        delta = delta.type(torch.float64)
        lambd = lambd.type(torch.float64)

        return delta.detach(), lambd.detach()

    def _dual_function(self, var, delta, lambd):
        eta, theta = var[0], var[1:]
        epsilon = self.max_kl_divergence
        g = eta * (epsilon + torch.log(torch.mean(torch.exp(delta / eta))))
        g += self.critic_reg * np.sum(theta ** 2)

        return g

    def _dual_function_grad(self, var, delta, lambd):
        eta, theta = var[0], var[1:]
        epsilon = self.max_kl_divergence
        weights = torch.exp(epsilon + delta / eta)
        del_theta = eta * (torch.sum(weights * lambd, dim=0) / torch.sum(weights))
        del_eta = torch.log(torch.sum(weights)) - (1 / eta**2) * (torch.sum(weights * delta) / torch.sum(weights))
        del_g = torch.hstack([del_theta, del_eta])

        return del_g

    def _update_policy(self, states, actions, rewards, next_states):
        # Update policy by weighted maximum likelihood estimate
        delta, _ = self._compute_weights(states, rewards, next_states)
        weights = torch.diag(torch.exp(delta / self.eta.detach()).squeeze())

        phi_a = self.actor_net(states).detach().type(torch.float64)
        actions = actions.type(torch.float64)
        self.omega = torch.inverse(phi_a.T @ weights @ phi_a + self.actor_reg * np.eye(self.rbf_dim)) @ phi_a.T @ weights @ actions

        diff = actions - phi_a @ self.omega
        std = torch.sum(weights @ (diff * diff), dim=0) / torch.sum(weights)
        self.std = std.reshape(1, self.a_dim)
