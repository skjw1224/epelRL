import os
import numpy as np
import scipy.optimize as optim

from .base_algorithm import Algorithm
from network.rbf import RBF
from utility.buffer import RolloutBuffer


class REPS(Algorithm):
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
        self.critic_reg = self.config['critic_reg']
        self.actor_reg = self.config['actor_reg']
        self.max_kl_divergence = self.config['max_kl_divergence']

        config['buffer_size'] = self.nT * self.num_rollout
        config['batch_size'] = self.nT * self.num_rollout
        self.rollout_buffer = RolloutBuffer(config)

        # Critic network
        self.critic = RBF(self.s_dim, self.rbf_dim, self.rbf_type)
        self.eta = np.random.randn(1, 1)
        self.theta = np.random.randn(self.rbf_dim, 1)

        # Actor network
        self.actor = RBF(self.s_dim, self.rbf_dim, self.rbf_type)
        self.omega = np.random.randn(self.rbf_dim, self.a_dim)
        self.actor_std = np.ones((1, self.a_dim))

        self.loss_lst = ['Critic loss', 'Actor loss']

    def ctrl(self, state):
        phi = self.actor.forward(state.T)
        mean = phi @ self.omega
        action = np.random.normal(mean, self.actor_std)
        action = np.clip(action.T, -1., 1.)

        return action

    def add_experience(self, experience):
        self.rollout_buffer.add(experience)

    def train(self):
        # Replay buffer sample
        sample = self.rollout_buffer.sample(use_tensor=False)
        states = sample['states']
        actions = sample['actions']
        rewards = sample['rewards']
        next_states = sample['next_states']

        # Update critic (value function)
        critic_loss = self._critic_update(states, rewards, next_states)

        # Update actor (weighted maximum likelihood estimation)
        actor_loss = self._actor_update(states, actions, rewards, next_states)

        loss = np.array([critic_loss, actor_loss])

        return loss

    def _critic_update(self, states, rewards, next_states):
        # Compute dual function and the dual function's derivative
        g = self._dual_function
        del_g = self._dual_function_grad

        # Optimize value function
        x0 = np.concatenate([self.theta, self.eta]).squeeze()
        sol = optim.minimize(g, x0, method='L-BFGS-B', jac=del_g, args=(states, rewards, next_states),
                             bounds=(*((-np.inf, np.inf),) * self.rbf_dim, (1e-16, 1e16)))
                             # options={"iprint": 5})
        self.theta, self.eta = sol.x[:-1].reshape(-1, 1), sol.x[-1].reshape(-1, 1)
        critic_loss = sol.fun
        # print(f"critic_update_sucess: {sol.success}")
        return critic_loss

    def _compute_weights(self, states, rewards, next_states, theta, eta):
        phi_s = self.critic.forward(states)
        phi_s2 = self.critic.forward(next_states)

        # Compute Bellman error
        delta = -rewards + np.matmul(phi_s2, theta) - np.matmul(phi_s, theta)
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
        del_theta = np.sum(np.broadcast_to(weights, lambd.shape) * lambd, axis=0) / np.sum(weights)
        del_theta += 2 * self.critic_reg * theta
        del_eta = epsilon + np.log(np.mean(weights, axis=0)) - np.sum(weights * delta) / (eta * np.sum(weights))
        del_g = np.hstack((del_theta, del_eta))

        return del_g

    def _actor_update(self, states, actions, rewards, next_states):
        # Update policy by weighted maximum likelihood estimate
        delta, _, weights = self._compute_weights(states, rewards, next_states, self.theta, self.eta)
        weights = np.diag(weights.squeeze())
        phi_a = self.actor.forward(states)

        self.omega = np.linalg.inv(phi_a.T @ weights @ phi_a + self.actor_reg * np.eye(self.rbf_dim)) @ phi_a.T @ weights @ actions

        diff = actions - phi_a @ self.omega
        std = np.sum(weights @ (diff * diff), axis=0) / np.sum(weights)
        self.actor_std = std.reshape(1, self.a_dim)

        # Compute actor loss (weighted log likelihood)
        mean = phi_a @ self.omega
        std = np.broadcast_to(std, mean.shape)
        weights = np.diag(weights).reshape(-1, 1)
        log_determinant = -0.5 * (self.a_dim * np.log(2*np.pi)) + np.sum(np.log(std), axis=-1, keepdims=True)
        log_exp = -0.5 * np.sum((actions-mean)*(actions-mean) / std, axis=-1, keepdims=True)
        log_likelihood = log_determinant + log_exp
        actor_loss = - weights.T @ log_likelihood

        return actor_loss.item()

    def save(self, path, file_name):
        np.save(os.path.join(path, '_critic_centers.npy'), self.critic.centers)
        np.save(os.path.join(path, '_critic_shape_params.npy'), self.critic.shape_params)
        np.save(os.path.join(path, '_eta.npy'), self.eta)
        np.save(os.path.join(path, '_theta.npy'), self.theta)

        np.save(os.path.join(path, '_actor_centers.npy'), self.actor.centers)
        np.save(os.path.join(path, '_actor_shape_params.npy'), self.actor.shape_params)
        np.save(os.path.join(path, '_omega.npy'), self.omega)
        np.save(os.path.join(path, '_actor_std.npy'), self.actor_std)

    def load(self, path, file_name):
        self.critic.centers = np.load(os.path.join(path, '_critic_centers.npy'))
        self.critic.shape_params = np.load(os.path.join(path, '_critic_shape_params.npy'))
        self.eta = np.load(os.path.join(path, '_eta.npy'))
        self.theta = np.load(os.path.join(path, '_theta.npy'))

        # Actor network
        self.actor.centers = np.load(os.path.join(path, '_actor_centers.npy'))
        self.actor.shape_params = np.load(os.path.join(path, '_actor_shape_params.npy'))
        self.omega = np.load(os.path.join(path, '_omega.npy'))
        self.actor_std = np.load(os.path.join(path, '_actor_std.npy'))
