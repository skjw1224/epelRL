import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

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
GRAD_CLIP = 5.0

INITIAL_POLICY_INDEX = 5
AC_PE_TRAINING_INDEX = 10

class Qnetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a deep Q-learning network
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(Qnetwork, self).__init__()
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
        x = self.fc4(x)
        return x

class DQN(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.env_a_dim = self.env.a_dim
        self.a_dim = len(self.config.algorithm['action_mesh_idx'][0])


        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=BUFFER_SIZE, batch_size=MINIBATCH_SIZE)
        self.exp_noise = OU_Noise(size=self.env_a_dim)
        self.initial_ctrl = InitialControl(self.env, self.device)

        self.q_net = Qnetwork(self.s_dim, self.a_dim).to(self.device)  # s --> a
        self.target_q_net = Qnetwork(self.s_dim, self.a_dim).to(self.device) # s --> a

        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.q_net_opt = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS, weight_decay=L2REG)


    def ctrl(self, epi, step, x, u):
        u_idx = self.choose_action(epi, step, x, u)
        return u_idx

    def choose_action(self, epi, step, x, u):
        self.q_net.eval()
        with torch.no_grad():
            u_idx = self.q_net(x).min(-1)[1].unsqueeze(1) # (B, A)
        self.q_net.train()

        if step == 0: self.exp_schedule(epi)
        if random.random() <= self.epsilon and epi <= AC_PE_TRAINING_INDEX:
            u_idx = np.randint(self.a_dim, [1, 1]) # (B, A)

        return u_idx

    def add_experience(self, *single_expr):
        x, u_idx, r, x2, term = single_expr
        self.replay_buffer.add(*[x, u_idx, r, x2, term])

    def exp_schedule(self, epi):
        self.epsilon = EPSILON / (1. + (epi / EPI_DENOM))

    def train(self):
        s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample()

        q_batch = self.q_net(s_batch).gather(1, a_batch.long())

        q2_batch = self.target_q_net(s2_batch).detach().min(-1)[0].unsqueeze(1) * (~term_batch)
        q_target_batch = r_batch + q2_batch

        q_loss = F.mse_loss(q_batch, q_target_batch)

        self.q_net_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.q_net_opt.step()

        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(TAU * from_model.data + (1 - TAU) * to_model.data)

class InitialControl(object):
    def __init__(self, env, device):
        from pid import PID
        self.pid = PID(env, device)

    def controller(self, epi, step, x, u):
        return self.pid.ctrl(epi, step, x, u)



