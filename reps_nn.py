import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from replay_buffer import ReplayBuffer


class REPS_NN(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        # Hyperparameters
        self.h_nodes = self.config.hyperparameters['hidden_nodes']
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.critic_learning_rate = self.config.hyperparameters['critic_learning_rate']
        self.actor_learning_rate = self.config.hyperparameters['actor_learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.max_kl_divergence = self.config.hyperparameters['max_kl_divergence']
        self.batch_epi = self.config.hyperparameters['batch_epi']
        self.num_critic_update = self.config.hyperparameters['num_critic_update']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT*self.batch_epi, batch_size=self.nT*self.batch_epi)

        # Critic network
        self.critic_net = self.approximator(self.s_dim, 1, self.h_nodes).to(self.device)
        self.critic_net.eta = nn.Parameter(torch.rand([1])).to(self.device)
        self.critic_net_opt = optim.Adam(self.critic_net.parameters(), lr=self.critic_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor network
        self.actor_net = self.approximator(self.s_dim, self.a_dim, self.h_nodes).to(self.device)
        self.actor_net.log_std = nn.Parameter(torch.zeros(1, self.a_dim).to(self.device))
        self.actor_net_opt = optim.Adam(self.actor_net.parameters(), lr=self.actor_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

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

        self.actor_net.eval()
        with torch.no_grad():
            mean = self.actor_net(s)
            log_std = self.actor_net.log_std.expand_as(mean)
            std = torch.exp(log_std)
        self.actor_net.train()

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
            t, s, _, a = self.env.reset()
            for i in range(self.nT):
                a = self.ctrl(epi, i, s, a)
                t2, s2, _, r, is_term, _ = self.env.step(t, s, a)
                self.replay_buffer.add(*[s, a, r, s2, is_term])
                t, s = t2, s2

    def train(self):
        # Replay buffer sample
        s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample_sequence()

        # Update critic network
        critic_loss = self._critic_update(s_batch, r_batch, s2_batch)

        # Policy update
        actor_loss = self._actor_update(s_batch, a_batch, r_batch, s2_batch)

    def _critic_update(self, states, rewards, next_states):
        # Update critic network
        critic_loss = 0.
        for _ in range(self.num_critic_update):
            with torch.no_grad():
                target_value = rewards + self.critic_net(next_states).detach()
            delta = target_value - self.critic_net(states)
            max_delta = torch.max(delta)
            weights = torch.exp((delta - max_delta) / self.critic_net.eta)

            loss = self.critic_net.eta * (self.max_kl_divergence + torch.log(torch.mean(weights))) + max_delta
            self.critic_net_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.grad_clip_mag)
            self.critic_net_opt.step()

            critic_loss += loss.detach().cpu().item()

        critic_loss /= self.num_critic_update

        return critic_loss

    def _actor_update(self, states, actions, rewards, next_states):
        # Update actor network
        with torch.no_grad():
            delta = rewards + self.critic_net(next_states).detach() - self.critic_net(states).detach()
            max_delta = torch.max(delta)
            weights = torch.exp((delta - max_delta) / self.critic_net.eta.detach())

        mean = self.actor_net(states)
        std = torch.exp(self.actor_net.log_std.expand_as(mean))
        log_determinant = -0.5 * (self.a_dim*np.log(2*np.pi) + torch.sum(torch.log(std), dim=-1, keepdim=True))
        log_exp = -0.5 * torch.sum((actions - mean) * (actions - mean) / std, dim=-1, keepdim=True)
        log_likelihood = log_determinant + log_exp

        loss = - torch.dot(weights.squeeze(), log_likelihood.squeeze())
        self.actor_net_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.grad_clip_mag)
        self.actor_net_opt.step()

        actor_loss = loss.detach().cpu().item()

        return actor_loss

