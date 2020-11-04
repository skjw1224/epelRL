import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import time

from nn_create import NeuralNetworks
from replay_buffer import ReplayBuffer
from explorers import OU_Noise

class GDHP(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim

        # hyperparameters
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.explore_epi_idx = self.config.hyperparameters['explore_epi_idx']
        self.buffer_size = self.config.hyperparameters['buffer_size']
        self.minibatch_size = self.config.hyperparameters['minibatch_size']
        self.crt_h_nodes = self.config.hyperparameters['hidden_nodes']
        self.act_h_nodes = self.config.hyperparameters['hidden_nodes']
        self.cos_h_nodes = self.config.hyperparameters['hidden_nodes']
        self.crt_learning_rate = self.config.hyperparameters['critic_learning_rate']
        self.act_learning_rate = self.config.hyperparameters['actor_learning_rate']
        self.cst_learning_rate = self.config.hyperparameters['costate_learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.eps = self.config.hyperparameters['eps_greedy']
        self.epi_denom = self.config.hyperparameters['eps_greedy_denom']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.tau = self.config.hyperparameters['tau']

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.buffer_size)
        self.exp_noise = OU_Noise(size=self.a_dim)
        self.initial_ctrl = InitialControl(self.env, self.device)

        # Critic (+target) net
        self.critic_net = NeuralNetworks(self.s_dim, 1, self.crt_h_nodes).to(self.device)  # s --> 1
        self.target_critic_net = NeuralNetworks(self.s_dim, 1, self.crt_h_nodes).to(self.device) # s --> 1

        for to_model, from_model in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.critic_net_opt = torch.optim.Adam(self.critic_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor (+target) net
        self.actor_net = NeuralNetworks(self.s_dim, self.a_dim, self.act_h_nodes).to(self.device)  # s --> a
        self.target_actor_net = NeuralNetworks(self.s_dim, self.a_dim, self.act_h_nodes).to(self.device)  # s --> a

        for to_model, from_model in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.actor_net_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Costate (+target) net
        self.costate_net = NeuralNetworks(self.s_dim, self.s_dim, self.cos_h_nodes).to(self.device)  # s --> s
        self.target_costate_net = NeuralNetworks(self.s_dim, self.s_dim, self.cos_h_nodes).to(self.device)  # s --> s

        for to_model, from_model in zip(self.target_costate_net.parameters(), self.costate_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.costate_net_opt = torch.optim.Adam(self.costate_net.parameters(), lr=self.cst_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)


    def ctrl(self, epi, step, x, u):
        start_time = time.time()
        a_exp = self.exp_schedule(epi, step)
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.controller(epi, step, x, u)
            a_nom = torch.from_numpy(a_nom)
        elif self.init_ctrl_idx <= epi < self.explore_epi_idx:
            a_nom = self.choose_action(x)
        else:
            a_nom = self.choose_action(x)

        a_val = a_nom + torch.tensor(a_exp, dtype=torch.float, device=self.device)

        return a_val

    def choose_action(self, s):
        # Option: target_actor_net OR actor_net?
        self.target_actor_net.eval()
        with torch.no_grad():
            a_nom = self.target_actor_net(s)
        self.target_actor_net.train()
        a_nom.detach()
        return a_nom

    def exp_schedule(self, epi, step):
        noise = self.exp_noise.sample() / 10.
        if step == 0: self.epsilon = self.eps / (1. + (epi / self.epi_denom))
        a_exp = noise * self.epsilon
        return a_exp

    def add_experience(self, *single_expr):
        x, u, r, x2, term, derivs = single_expr
        # dfdx, dfdu, dcdx, d2cdu2_inv = derivs
        self.replay_buffer.add(*[x, u, r, x2, term, *derivs])

    def train(self, step):
        def nn_update_one_step(orig_net, target_net, opt, loss):
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(orig_net.parameters(), self.grad_clip_mag)
            opt.step()

            """Updates the target network in the direction of the local network but by taking a step size
           less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
            for to_model, from_model in zip(target_net.parameters(), orig_net.parameters()):
                to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        # Replay buffer sample
        x_batch, u_batch, r_batch, x2_batch, term_batch, dfdx_batch, dfdu_batch, dcdx_batch, d2cdu2inv_batch = self.replay_buffer.sample()

        # Critic Train
        q_batch = self.critic_net(x_batch)
        q2_batch = self.target_critic_net(x2_batch) * (1 - term_batch)
        q_target_batch = r_batch + q2_batch

        q_loss = F.mse_loss(q_batch, q_target_batch)

        nn_update_one_step(self.critic_net, self.target_critic_net, self.critic_net_opt, q_loss)

        # Costate Train
        l_batch = self.costate_net(x_batch)
        l2_batch = self.target_costate_net(x2_batch) * (1 - term_batch)
        l_target_batch = (dcdx_batch + l2_batch.unsqueeze(1) @ dfdx_batch).squeeze(1) # (B, S)

        l_loss = F.mse_loss(l_batch, l_target_batch)

        nn_update_one_step(self.costate_net, self.target_costate_net, self.costate_net_opt, l_loss)

        # Actor Train
        u_batch = self.actor_net(x_batch)
        a_target_batch = torch.clamp((-0.5 * l2_batch.detach().unsqueeze(1) @ dfdu_batch @ d2cdu2inv_batch), -1., 1.).squeeze(1)

        a_loss = F.mse_loss(u_batch, a_target_batch)

        nn_update_one_step(self.actor_net, self.target_actor_net, self.actor_net_opt, a_loss)

from ilqr import ILQR
class InitialControl(object):
    def __init__(self, env, device):
        self.ilqr = ILQR(env, device)

    def controller(self, epi, step, x, u):
        return self.ilqr.ctrl(epi, step, x, u)