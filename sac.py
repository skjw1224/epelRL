import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from replay_buffer import ReplayBuffer


class SAC(object):
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
        self.buffer_size = self.config.hyperparameters['buffer_size']
        self.minibatch_size = self.config.hyperparameters['minibatch_size']
        self.crt_learning_rate = self.config.hyperparameters['critic_learning_rate']
        self.act_learning_rate = self.config.hyperparameters['actor_learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.tau = self.config.hyperparameters['tau']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.minibatch_size)

        # Critic networks
        self.critic_net_a = self.approximator(self.s_dim + self.a_dim, 1, self.h_nodes).to(self.device)
        self.target_critic_net_a = self.approximator(self.s_dim + self.a_dim, 1, self.h_nodes).to(self.device)
        for to_model, from_model in zip(self.target_critic_net_a.parameters(), self.critic_net_a.parameters()):
            to_model.data.copy_(from_model.data.clone())
        self.critic_net_opt_a = optim.Adam(self.critic_net_a.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.critic_net_b = self.approximator(self.s_dim + self.a_dim, 1, self.h_nodes).to(self.device)
        self.target_critic_net_b = self.approximator(self.s_dim + self.a_dim, 1, self.h_nodes).to(self.device)
        for to_model, from_model in zip(self.target_critic_net_b.parameters(), self.critic_net_b.parameters()):
            to_model.data.copy_(from_model.data.clone())
        self.critic_net_opt_b = optim.Adam(self.critic_net_b.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor net
        self.actor_net = self.approximator(self.s_dim, 2 * self.a_dim, self.h_nodes).to(self.device)
        self.actor_net_opt = optim.Adam(self.actor_net.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Temperature learning
        self.automatic_temp_tuning = self.config.hyperparameters['automatic_temp_tuning']
        if self.automatic_temp_tuning:
            self.target_entropy = - self.a_dim
            self.log_temp = torch.zeros(1, requires_grad=True, device=self.device)
            self.temp = self.log_temp.exp()
            self.temp_optim = optim.Adam([self.log_temp], lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg) # 혹은 따로 지정
        else:
            self.temp = self.config.hyperparameters['temperature']

        self.loss_lst = ['Critic_a_loss', 'Critic_b_loss', 'Actor_loss']

    def ctrl(self, epi, step, s, a):
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.ctrl(epi, step, s, a)
            a_val = self.explorer.sample(epi, step, a_nom)
        else:
            a_val = self._choose_action(s)

        a_val = np.clip(a_val, -1., 1.)

        return a_val

    def add_experience(self, *single_expr):
        s, a, r, s2, is_term = single_expr
        self.replay_buffer.add(*[s, a, r, s2, is_term])

    def _choose_action(self, s):
        # numpy to torch
        s = torch.from_numpy(s.T).float().to(self.device)

        self.actor_net.eval()
        with torch.no_grad():
            a_pred = self.actor_net(s)
        self.actor_net.train()

        mean, log_std = a_pred[:, :self.a_dim], a_pred[:, self.a_dim:]
        std = torch.exp(log_std)
        a_distribution = Normal(mean, std)
        a = a_distribution.rsample()  # Reparameterization trick
        a = torch.tanh(a)

        # torch to numpy
        a = a.T.detach().cpu().numpy()

        return a

    def train(self):
        # Replay buffer sample
        s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample()

        # Critic network updat
        critic_loss_a, critic_loss_b = self._critic_update(s_batch, a_batch, r_batch, s2_batch)

        # Actor network update
        actor_loss = self._actor_update(s_batch)

        critic_loss_a = critic_loss_a.detach().cpu().item()
        critic_loss_b = critic_loss_b.detach().cpu().item()
        actor_loss = actor_loss.detach().cpu().item()
        loss = np.array([critic_loss_a, critic_loss_b, actor_loss])

        return loss

    def _get_log_prob(self, s_batch):
        a_pred = self.actor_net(s_batch)
        mean, log_std = a_pred[:, :self.a_dim], a_pred[:, self.a_dim:]
        std = torch.exp(log_std)
        distribution = Normal(mean, std)
        a_batch = distribution.rsample()
        a_batch = torch.tanh(a_batch)
        log_prob = distribution.log_prob(a_batch)
        log_prob -= torch.log(1 - a_batch.pow(2) + 1e-7)  # tanh에 따른 미분 값 보정, epsilon은 -inf 방지용
        log_prob = log_prob.sum(1, keepdim=True)

        return a_batch, log_prob

    def _critic_update(self, s_batch, a_batch, r_batch, s2_batch):
        with torch.no_grad():
            a2_batch, log_pi = self._get_log_prob(s2_batch)
            q2_batch_a = self.target_critic_net_a(torch.cat([s2_batch, a2_batch], dim=-1)).detach()
            q2_batch_b = self.target_critic_net_b(torch.cat([s2_batch, a2_batch], dim=-1)).detach()
            max_q2_batch = torch.max(q2_batch_a, q2_batch_b) - self.temp * log_pi
            q_target_batch = r_batch + max_q2_batch

        q_batch_a = self.critic_net_a(torch.cat([s_batch, a_batch], dim=-1))
        q_batch_b = self.critic_net_b(torch.cat([s_batch, a_batch], dim=-1))
        critic_loss_a = F.mse_loss(q_batch_a, q_target_batch)
        critic_loss_b = F.mse_loss(q_batch_b, q_target_batch)

        self.nn_update(self.critic_net_a, self.target_critic_net_a, self.critic_net_opt_a, critic_loss_a)
        self.nn_update(self.critic_net_b, self.target_critic_net_b, self.critic_net_opt_b, critic_loss_b)

        return critic_loss_a, critic_loss_b

    def _actor_update(self, s_batch):
        a_batch, log_pi = self._get_log_prob(s_batch)
        q_batch_a = self.critic_net_a(torch.cat([s_batch, a_batch], dim=-1))
        q_batch_b = self.critic_net_b(torch.cat([s_batch, a_batch], dim=-1))
        actor_loss = (torch.max(q_batch_a, q_batch_b) - (self.temp * log_pi)).mean()

        self.nn_update(self.actor_net, None, self.actor_net_opt, actor_loss)

        if self.automatic_temp_tuning:
            self._temperature_adjustment(log_pi)

        return actor_loss

    def _temperature_adjustment(self, log_pi):
        temp_loss = -(self.log_temp * (log_pi + self.target_entropy).detach()).mean()  # TODO: Check the sign
        self.nn_update(None, None, self.temp_optim, temp_loss)
        self.temp = self.log_temp.exp()

    def nn_update(self, orig_net, target_net, opt, loss):
        opt.zero_grad()
        loss.backward()
        if orig_net is not None:
            torch.nn.utils.clip_grad_norm_(orig_net.parameters(), self.grad_clip_mag)
        opt.step()

        if target_net is not None:
            for to_model, from_model in zip(target_net.parameters(), orig_net.parameters()):
                to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)
