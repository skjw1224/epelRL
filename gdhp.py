import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import time

from replay_buffer import ReplayBuffer
from ou_noise import OU_Noise

BUFFER_SIZE = 600
MINIBATCH_SIZE = 32
TAU = 0.05
EPSILON = 0.1
EPI_DENOM = 1.

LEARNING_RATE = 2E-4
ADAM_EPS = 1E-4
L2REG = 1E-3
GRAD_CLIP = 10.0

INITIAL_POLICY_INDEX = 10
AC_PE_TRAINING_INDEX = 10

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a deep Q-learning network
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(Network, self).__init__()
        n_h_nodes = [50, 50, 30]

        self.fc1 = nn.Linear(input_dim, n_h_nodes[0])
        self.bn1 = nn.BatchNorm1d(n_h_nodes[0])
        self.fc2 = nn.Linear(n_h_nodes[0], n_h_nodes[1])
        self.bn2 = nn.BatchNorm1d(n_h_nodes[1])
        self.fc3 = nn.Linear(n_h_nodes[1], n_h_nodes[2])
        self.bn3 = nn.BatchNorm1d(n_h_nodes[2])
        self.fc4 = nn.Linear(n_h_nodes[2], output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        # x = F.leaky_relu(self.bn1(self.fc1(x)))
        # x = F.leaky_relu(self.bn2(self.fc2(x)))
        # x = F.leaky_relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

class GDHP(object):
    def __init__(self, env, device):
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim

        self.device = device

        self.replay_buffer = ReplayBuffer(env, device, buffer_size=BUFFER_SIZE, batch_size=MINIBATCH_SIZE)
        self.exp_noise = OU_Noise(size=self.a_dim)
        self.initial_ctrl = InitialControl(env, device)

        # Critic (+target) net
        self.critic_net = Network(self.s_dim, 1).to(device)  # s --> 1
        self.target_critic_net = Network(self.s_dim, 1).to(device) # s --> 1

        for to_model, from_model in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.critic_net_opt = torch.optim.Adam(self.critic_net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS, weight_decay=L2REG)

        # Actor (+target) net
        self.actor_net = Network(self.s_dim, self.a_dim).to(device)  # s --> a
        self.target_actor_net = Network(self.s_dim, self.a_dim).to(device)  # s --> a

        for to_model, from_model in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.actor_net_opt = torch.optim.Adam(self.actor_net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS, weight_decay=L2REG)

        # Costate (+target) net
        self.costate_net = Network(self.s_dim, self.s_dim).to(device)  # s --> s
        self.target_costate_net = Network(self.s_dim, self.s_dim).to(device)  # s --> s

        for to_model, from_model in zip(self.target_costate_net.parameters(), self.costate_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.costate_net_opt = torch.optim.Adam(self.costate_net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS, weight_decay=L2REG)


    def ctrl(self, epi, step, x, u):

        start_time = time.time()
        a_exp = self.exp_schedule(epi, step)
        if epi < INITIAL_POLICY_INDEX:
            a_nom = self.initial_ctrl.controller(epi, step, x, u)
            a_nom = torch.from_numpy(a_nom)
        elif INITIAL_POLICY_INDEX <= epi < AC_PE_TRAINING_INDEX:
            a_nom = self.choose_action(x)
        else:
            a_nom = self.choose_action(x)

        if INITIAL_POLICY_INDEX <= epi:
            print("time:", time.time() - start_time)

        start_time = time.time()

        a_val = a_nom + torch.tensor(a_exp, dtype=torch.float, device=self.device)
        if epi>= 1:
            self.nn_train()
            print("train time:", time.time() - start_time)
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
        if step == 0: self.epsilon = EPSILON / (1. + (epi / EPI_DENOM))
        a_exp = noise * self.epsilon
        return a_exp

    def add_experience(self, *single_expr):
        x, u, r, x2, term, derivs = single_expr
        # dfdx, dfdu, dcdx, d2cdu2_inv = derivs
        self.replay_buffer.add(*[x, u, r, x2, term, *derivs])

    def nn_train(self):

        def nn_update_one_step(orig_net, target_net, opt, loss):
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(orig_net.parameters(), GRAD_CLIP)
            opt.step()

            """Updates the target network in the direction of the local network but by taking a step size
           less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
            for to_model, from_model in zip(target_net.parameters(), orig_net.parameters()):
                to_model.data.copy_(TAU * from_model.data + (1 - TAU) * to_model.data)

        # Replay buffer sample
        s_batch, a_batch, r_batch, s2_batch, term_batch, dfdx_batch, dfdu_batch, dcdx_batch, d2cdu2inv_batch = self.replay_buffer.sample()

        # Critic Train
        q_batch = self.critic_net(s_batch)
        q2_batch = self.target_critic_net(s2_batch) * (~term_batch)
        q_target_batch = r_batch + q2_batch

        q_loss = F.mse_loss(q_batch, q_target_batch)

        nn_update_one_step(self.critic_net, self.target_critic_net, self.critic_net_opt, q_loss)

        # Costate Train
        l_batch = self.costate_net(s_batch)
        l2_batch = self.target_costate_net(s2_batch) * (~term_batch)
        l_target_batch = (dcdx_batch + l2_batch.unsqueeze(1) @ dfdx_batch).squeeze(1) # (B, S)

        l_loss = F.mse_loss(l_batch, l_target_batch)

        nn_update_one_step(self.costate_net, self.target_costate_net, self.costate_net_opt, l_loss)

        # Actor Train
        a_batch = self.actor_net(s_batch)
        a_target_batch = torch.clamp((-0.5 * l2_batch.detach().unsqueeze(1) @ dfdu_batch @ d2cdu2inv_batch), -1., 1.).squeeze(1)

        a_loss = F.mse_loss(a_batch, a_target_batch)

        nn_update_one_step(self.actor_net, self.target_actor_net, self.actor_net_opt, a_loss)

from ilqr import Ilqr
class InitialControl(object):
    def __init__(self, env, device):
        self.ilqr = Ilqr(env, device)

    def controller(self, epi, step, x, u):
        return self.ilqr.ctrl(epi, step, x, u)