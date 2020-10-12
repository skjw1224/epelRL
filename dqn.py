import torch

import torch.nn.functional as F
import numpy as np
import random

from nn_create import NeuralNetworks
from replay_buffer import ReplayBuffer
from ou_noise import OU_Noise

class DQN(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.env_a_dim = self.env.a_dim
        self.a_dim = len(self.config.algorithm['action_mesh_idx'][0])

        # hyperparameters
        self.explore_epi_idx = self.config.hyperparameters['explore_epi_idx']
        self.buffer_size = self.config.hyperparameters['buffer_size']
        self.minibatch_size = self.config.hyperparameters['minibatch_size']
        self.learning_rate = self.config.hyperparameters['learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.eps = self.config.hyperparameters['eps_greedy']
        self.epi_denom = self.config.hyperparameters['eps_greedy_denom']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.tau = self.config.hyperparameters['tau']


        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.minibatch_size)
        self.exp_noise = OU_Noise(size=self.env_a_dim)
        self.initial_ctrl = InitialControl(self.config)

        self.q_net = NeuralNetworks(self.s_dim, self.a_dim).to(self.device)  # s --> a
        self.target_q_net = NeuralNetworks(self.s_dim, self.a_dim).to(self.device) # s --> a

        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.q_net_opt = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

    def ctrl(self, epi, step, x, u):
        if epi < self.explore_epi_idx:
            if step == 0: self.exp_schedule(epi)
            if np.random.random() <= self.epsilon:
                u_idx = np.randint(self.a_dim, [1, 1]) # (B, A)
        else:
            u_idx = self.choose_action(epi, step, x, u)
        return u_idx

    def choose_action(self, epi, step, x, u):
        self.q_net.eval()
        with torch.no_grad():
            u_idx = self.q_net(x).min(-1)[1].unsqueeze(1) # (B, A)
        self.q_net.train()
        return u_idx

    def add_experience(self, *single_expr):
        x, u_idx, r, x2, term = single_expr
        self.replay_buffer.add(*[x, u_idx, r, x2, term])

    def exp_schedule(self, epi):
        self.epsilon = self.eps / (1. + (epi / self.epi_denom))

    def train(self, step):
        x_batch, u_batch, r_batch, x2_batch, term_batch = self.replay_buffer.sample()

        q_batch = self.q_net(x_batch).gather(1, u_batch.long())

        q2_batch = self.target_q_net(x2_batch).detach().min(-1)[0].unsqueeze(1) * (~term_batch)
        q_target_batch = r_batch + q2_batch

        q_loss = F.mse_loss(q_batch, q_target_batch)

        self.q_net_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip_mag)
        self.q_net_opt.step()

        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""

        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        return q_loss

class InitialControl(object):
    def __init__(self, config):
        from pid import PID
        self.pid = PID(config.env, config.device)

    def controller(self, epi, step, x, u):
        return self.pid.ctrl(epi, step, x, u)



