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

        # Critic network
        self.critic_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type)
        self.eta = np.random.rand(1, 1)
        self.theta = np.random.rand(self.rbf_dim, 1)

        # Actor network
        self.actor_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type)
        self.actor_std = torch.zeros(1, self.a_dim)
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

        phi = self.actor_net(s)
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

        # Update critic (value function)
        critic_loss = self._critic_update(s_batch, r_batch, s2_batch)

        # Update actor (weighted maximum likelihood estimation)
        actor_loss = self._actor_update(s_batch, a_batch, r_batch, s2_batch)

    def _critic_update(self, states, rewards, next_states):
        # Compute dual function and the dual function's derivative
        g = self._dual_function
        del_g = self._dual_function_grad

        # Optimize value function
        x0 = np.concatenate([self.theta, self.eta])
        sol = optim.minimize(g, x0, method='L-BFGS-B', jac=del_g, args=(states, rewards, next_states),
                             bounds=((1e-16, 1e16),) + ((-np.inf, np.inf),) * self.rbf_dim)
        self.theta, self.eta = sol.x[:-1].reshape(-1, 1), sol.x[-1].reshape(-1, 1)
        critic_loss = sol.fun

        return critic_loss

    def _compute_weights(self, states, rewards, next_states, theta, eta):
        phi_s = self.critic_net(states).detach().numpy()
        phi_s2 = self.critic_net(next_states).detach().numpy()
        rewards = rewards.detach().numpy()

        # Compute Bellman error
        delta = rewards + np.matmul(phi_s2, theta) - np.matmul(phi_s, theta)
        delta = delta - np.max(delta)

        # Compute feature difference
        lambd = phi_s2 - phi_s

        # Compute weights
        weights = np.exp(delta / eta)

        return delta, lambd, weights

    def _dual_function(self, var, states, rewards, next_states):
        theta, eta = var[:-1], var[-1]
        epsilon = self.max_kl_divergence

        _, _, weights = self._compute_weights(states, rewards, next_states, theta.reshape(-1, 1), eta.reshape(-1, 1))
        g = eta * (epsilon + np.log(np.mean(weights, axis=0)))
        g += self.critic_reg * np.sum(theta ** 2)

        return g.item()

    def _dual_function_grad(self, var, states, rewards, next_states):
        theta, eta = var[:-1], var[-1]
        epsilon = self.max_kl_divergence

        delta, lambd, weights = self._compute_weights(states, rewards, next_states, theta.reshape(-1, 1), eta.reshape(-1, 1))
        del_theta = eta * (np.sum(np.broadcast_to(weights, lambd.shape) * lambd, axis=0) / np.sum(weights))
        del_eta = epsilon + np.log(np.mean(weights, axis=0)) - np.sum(weights * delta) / (eta**2 * np.sum(weights))
        del_g = np.hstack((del_theta, del_eta))

        return del_g

    def _actor_update(self, states, actions, rewards, next_states):
        # Update policy by weighted maximum likelihood estimate
        delta, _, weights = self._compute_weights(states, rewards, next_states, self.theta, self.eta)
        weights = np.diag(weights.squeeze())
        phi_a = self.actor_net(states).detach().numpy()
        actions = actions.detach().numpy()
        omega = np.linalg.inv(phi_a.T @ weights @ phi_a + self.actor_reg * np.eye(self.rbf_dim)) @ phi_a.T @ weights @ actions
        self.omega = torch.from_numpy(omega).float()

        diff = actions - phi_a @ omega
        std = np.sum(weights @ (diff * diff), axis=0) / np.sum(weights)
        self.actor_std = torch.from_numpy(std.reshape(1, self.a_dim)).float()

        # Compute actor loss (weighted log likelihood)
        mean = phi_a @ omega
        std = np.broadcast_to(std, mean.shape)
        weights = np.diag(weights).reshape(-1, 1)
        log_determinant = -0.5 * (self.a_dim * np.log(2*np.pi)) + np.sum(np.log(std), axis=-1, keepdims=True)
        log_exp = -0.5 * np.sum((actions-mean)*(actions-mean) / std, axis=-1, keepdims=True)
        log_likelihood = log_determinant + log_exp
        actor_loss = - weights.T @ log_likelihood

        return actor_loss.item()
