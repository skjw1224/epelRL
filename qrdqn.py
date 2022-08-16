import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
my_CSTR = os.getcwd()

from pid import PID
from replay_buffer import ReplayBuffer
from explorers import OU_Noise
from dqn import DQN

class QRDQN(DQN):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.env_a_dim = self.env.a_dim
        self.a_dim = self.config.algorithm['controller']['action_mesh_idx'][-1]

        # hyperparameters
        self.h_nodes = self.config.hyperparameters['hidden_nodes']
        self.explore_epi_idx = self.config.hyperparameters['explore_epi_idx']
        self.buffer_size = self.config.hyperparameters['buffer_size']
        self.minibatch_size = self.config.hyperparameters['minibatch_size']
        self.learning_rate = self.config.hyperparameters['learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.tau = self.config.hyperparameters['tau']
        self.n_quantiles = self.config.hyperparameters['n_quantiles']
        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.minibatch_size)

        self.quantile_taus = ((2 * torch.arange(self.n_quantiles) + 1) / (2. * self.n_quantiles)).unsqueeze(0).to(self.device)

        # q network
        self.q_net = self.approximator(self.s_dim, self.a_dim * self.n_quantiles, self.h_nodes).to(self.device)  # s --> a
        self.target_q_net = self.approximator(self.s_dim, self.a_dim * self.n_quantiles, self.h_nodes).to(self.device) # s --> a

        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        # TODO: RMSprop vs. Adam
        self.q_net_opt = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.prev_a_idx = None

    def choose_action(self, epi, step, x, u):
        # Numpy to torch
        x = torch.from_numpy(x.T).float().to(self.device) # (B, 1)
        self.q_net.eval()
        with torch.no_grad():
            u_idx = self.get_value_distribution(self.q_net, x).mean(2).min(1)[1].unsqueeze(1)
        self.q_net.train()

        # Torch to Numpy
        u_idx = u_idx.cpu().detach().numpy()
        return u_idx

    def train(self, step):
        if len(self.replay_buffer) > 0:
            x_batch, u_batch, r_batch, x2_batch, term_batch = self.replay_buffer.sample()

            q_distribution = self.get_value_distribution(self.q_net, x_batch)
            q_batch = q_distribution.gather(1, u_batch.unsqueeze(-1).repeat(1, 1, self.n_quantiles).long())

            q2_distribution = self.get_value_distribution(self.target_q_net, x2_batch, False)
            u_max_idx_batch = q2_distribution.mean(2).min(1)[1].unsqueeze(1)
            q2_batch = q2_distribution.gather(1, u_max_idx_batch.unsqueeze(-1).repeat(1, 1, self.n_quantiles).long())
            q_target_batch = r_batch.unsqueeze(2) + -(-1 + term_batch.unsqueeze(2).float()) * q2_batch

            q_loss = self.quantile_huber_loss(q_batch, q_target_batch)

            self.q_net_opt.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_mag)
            self.q_net_opt.step()

            for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

            q_loss = q_loss.cpu().detach().numpy().item()
        else:
            q_loss = 0.

        return q_loss

    def get_value_distribution(self, net, s, stack_graph=True):
        if stack_graph:
            net_value = net(s)
        else:
            net_value = net(s).detach()
        return net_value.view(-1, self.a_dim, self.n_quantiles)

    def quantile_huber_loss(self, q_batch, q_target_batch):
        qh_loss_batch = torch.tensor(0., device=self.device)
        huber_loss_fnc = torch.nn.SmoothL1Loss(reduction='none')
        for n, q in enumerate(q_batch):
            q_target = q_target_batch[n]
            error = q_target - q.transpose(0,1)
            huber_loss = huber_loss_fnc(error, torch.zeros(error.shape, device=self.device))
            qh_loss = (huber_loss * (self.quantile_taus - (error < 0).float()).abs()).mean(1).sum(0)
            qh_loss_batch = qh_loss_batch + qh_loss
        return qh_loss_batch
