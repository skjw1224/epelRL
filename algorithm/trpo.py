import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np

from .base_algorithm import Algorithm
from network.network import ActorMlp, CriticMLP
from replay_buffer.replay_buffer import ReplayBuffer


class TRPO(Algorithm):
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.s_dim = self.config.s_dim
        self.a_dim = self.config.a_dim
        self.nT = self.config.nT

        # Hyperparameters
        self.num_hidden_nodes = self.config.num_hidden_nodes
        self.num_hidden_layers = self.config.num_hidden_layers
        hidden_dim_lst = [self.num_hidden_nodes for _ in range(self.num_hidden_layers)]

        self.gamma = self.config.gamma
        self.critic_lr = self.config.critic_lr
        self.actor_lr = self.config.actor_lr
        self.adam_eps = self.config.adam_eps
        self.l2_reg = self.config.l2_reg
        self.grad_clip_mag = self.config.grad_clip_mag

        self.gae_lambda = self.config.gae_lambda
        self.gae_gamma = self.config.gae_gamma
        self.num_critic_update = self.config.num_critic_update
        self.num_cg_iterations = self.config.num_cg_iterations
        self.num_line_search = self.config.num_line_search
        self.max_kl_divergence = self.config.max_kl_divergence

        config.buffer_size = self.nT
        config.batch_size = self.nT
        self.replay_buffer = ReplayBuffer(config)

        # Critic network
        self.critic = CriticMLP(self.s_dim, 1, hidden_dim_lst, F.silu).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor network
        self.actor = ActorMlp(self.s_dim, self.a_dim, hidden_dim_lst, F.silu).to(self.device)
        self.old_actor = ActorMlp(self.s_dim, self.a_dim, hidden_dim_lst, F.silu).to(self.device)

        self.loss_lst = ['Critic loss', 'Actor loss']

    def ctrl(self, state):
        with torch.no_grad():
            state = torch.tensor(state.T, dtype=torch.float32, device=self.device)
            action, _ = self.actor(state, deterministic=False, reparam_trick=False, return_log_prob=False)
        
        action = np.clip(action.T.cpu().numpy(), -1., 1.)

        return action

    def add_experience(self, *single_expr):
        state, action, reward, next_state, done = single_expr
        self.replay_buffer.add(*[state, action, reward, next_state, done])

    def train(self):
        # Replay buffer sample
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_sequence()

        # Compute returns and advantages
        returns, advantages = self._gae_estimation(states, rewards, dones)

        # Compute the gradient of surrgoate loss and kl divergence
        self.old_actor.load_state_dict(self.actor.state_dict())
        surrogate_loss, kl_div = self._compute_surrogate_loss(states, actions, advantages)
        loss_grad = self._flat_gradient(surrogate_loss, self.actor.parameters(), retain_graph=True)
        kl_div_grad = self._flat_gradient(kl_div, self.actor.parameters(), create_graph=True)

        # Update actor using conjugate gradient method and backtracking line search
        actor_loss = self._actor_update(kl_div_grad, loss_grad, states, actions, advantages, surrogate_loss)

        # Update critic network several steps with respect to returns
        critic_loss = self._critic_update(states, returns)

        loss = np.array([critic_loss, actor_loss])

        # Clear replay buffer
        self.replay_buffer.clear()

        return loss

    def _gae_estimation(self, states, rewards, terms):
        # Compute generalized advantage estimations (GAE) and returns
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        values = self.critic(states)

        for t in reversed(range(len(rewards))):
            prev_return = rewards[t] + self.gae_gamma * prev_return * (1 - terms[t])
            td_error = rewards[t] + self.gae_gamma * prev_value * (1 - terms[t]) - values.data[t]
            prev_advantage = td_error + self.gae_gamma * self.gae_lambda * prev_advantage * (1 - terms[t])
            prev_value = values.data[t]

            returns[t] = prev_return
            advantages[t] = prev_advantage

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # advantage normalization

        return returns, advantages

    def _compute_surrogate_loss(self, states, actions, advantages):
        # Compute surrogate loss and KL divergence
        log_probs_new, distribution_new = self._get_log_probs(states, actions, True)
        log_probs_old, distribution_old = self._get_log_probs(states, actions, False)

        surrogate_loss = (advantages * torch.exp(log_probs_new - log_probs_old.detach())).mean()
        kl_div = kl_divergence(distribution_old, distribution_new).mean()

        return surrogate_loss, kl_div

    def _get_log_probs(self, states, actions, is_new_actor):
        if is_new_actor:
            means = self.actor(states)
            log_stds = self.actor.log_std.expand_as(means)
            stds = torch.exp(log_stds)
        else:
            with torch.no_grad():
                means = self.actor(states)
                log_stds = self.actor.log_std.expand_as(means)
                stds = torch.exp(log_stds)
        distribution = Normal(means, stds)
        log_probs = distribution.log_prob(actions)

        return log_probs, distribution

    def _flat_gradient(self, tensor, parameters, retain_graph=False, create_graph=False):
        # Compute the gradient of tensor using pytorch autograd and flatten the gradient
        if create_graph:
            retain_graph = True

        grads = torch.autograd.grad(tensor, parameters, retain_graph=retain_graph, create_graph=create_graph)
        flat_grad = torch.cat([grad.view(-1) for grad in grads])

        return flat_grad

    def _actor_update(self, kl_div_grad, loss_grad, states, actions, advantages, surrogate_loss):
        # Compute a search direction using the conjugate gradient algorithm
        search_direction = self._conjugate_gradient(kl_div_grad, loss_grad.data)

        # Compute max step size and max step
        sHs = torch.matmul(search_direction, self._hessian_vector_product(kl_div_grad, search_direction))
        max_step_size = torch.sqrt(2 * self.max_kl_divergence / sHs)
        max_step = - max_step_size * search_direction

        # Do backtracking line search
        actor_loss = self._backtracking_line_search(max_step, states, actions, advantages, surrogate_loss)

        return actor_loss

    def _conjugate_gradient(self, kl_div_grad, loss_grad, residual_tol=1e-10):
        # Conjugate gradient method
        # From openai baseline code (https://github.com/openai/baselines/blob/master/baselines/common/cg.py)
        x = torch.zeros_like(loss_grad)
        r = loss_grad.clone()
        p = loss_grad.clone()
        rho = torch.dot(r, r)
        for _ in range(self.num_cg_iterations):
            hvp = self._hessian_vector_product(kl_div_grad, p)
            alpha = rho / torch.dot(p, hvp)
            x += alpha * p
            r -= alpha * hvp
            rho_new = torch.dot(r, r)
            p = r + (rho_new / rho) * p

            rho = rho_new
            if rho < residual_tol:
                break

        return x

    def _hessian_vector_product(self, kl_div_grad, vector, damping=1e-2):
        # Matrix-vector product between the Fisher information matrix and arbitrary vectors using direct method
        vp = torch.matmul(kl_div_grad, vector.detach())
        hvp = self._flat_gradient(vp, self.actor.parameters(), retain_graph=True)
        hvp += damping * vector

        return hvp

    def _backtracking_line_search(self, max_step, states, actions, advantages, prev_loss, fraction=0.9):
        for i in range(self.num_line_search):
            self._actor_params_update(max_step * (fraction ** i))
            with torch.no_grad():
                loss, kl_div = self._compute_surrogate_loss(states, actions, advantages)
            if loss < prev_loss and kl_div <= self.max_kl_divergence:
                break
            else:
                self._actor_params_update(- max_step)

        return loss.detach().cpu().item()

    def _actor_params_update(self, step):
        index = 0
        for params in self.actor.parameters():
            params_length = params.numel()
            params.data += step[index:index + params_length].view(params.shape)
            index += params_length

    def _critic_update(self, states, returns):
        criterion = nn.MSELoss(reduction='sum')
        arr = np.arange(self.nT)
        critic_loss = 0.

        for _ in range(self.num_critic_update):
            np.random.shuffle(arr)

            for i in range(self.nT // self.minibatch_size):
                batch_index = torch.LongTensor(arr[self.minibatch_size * i: self.minibatch_size * (i + 1)])
                values = self.critic(states[batch_index])
                target = returns[batch_index]

                loss = criterion(values, target)
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
                self.critic_optimizer.step()

                critic_loss += loss.detach().cpu().item()

        critic_loss /= self.nT * self.num_critic_update

        return critic_loss

    def save(self, path, file_name):
        torch.save(self.critic.state_dict(), os.path.join(path, file_name + '_critic.pt'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, file_name + '_critic_optimizer.pt'))

        torch.save(self.actor.state_dict(), os.path.join(path, file_name + '_actor.pt'))

    def load(self, path, file_name):
        self.critic.load_state_dict(torch.load(os.path.join(path, file_name + '_critic.pt')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic_optimizer.pt')))

        self.actor.load_state_dict(torch.load(os.path.join(path, file_name + '_actor.pt')))


