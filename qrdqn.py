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

NUM_QUANT = 21  # 200 -> 100 -> 51 -> 21
BUFFER_SIZE = 1600
MINIBATCH_SIZE = 32
# GAMMA = 0.99  # discount rate - not used
EPSILON = 0.8
EPI_DENOM = 0.3    # epel 10. edrcs 100.

RANDOM_SEED = 7318650  # for debugging, you can set any constant

TAU = 1E-5  # E-3 -> E-2 -> E-4
LEARNING_RATE = 1E-6
ADAM_EPS = 1E-2 / 32.
L2REG = 1E-3
GRAD_CLIP = 10.0
INITIAL_POLICY_INDEX = 5

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a deep Q-learning network
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DeepNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        n_h_nodes = [50, 50, 30]

        self.fc0 = nn.Linear(self.input_dim, n_h_nodes[0])
        self.bn0 = nn.BatchNorm1d(n_h_nodes[0])
        self.fc1 = nn.Linear(n_h_nodes[0], n_h_nodes[1])
        self.bn1 = nn.BatchNorm1d(n_h_nodes[1])
        self.fc2 = nn.Linear(n_h_nodes[1], n_h_nodes[2])
        self.bn2 = nn.BatchNorm1d(n_h_nodes[2])
        self.fc3 = nn.Linear(n_h_nodes[2], self.output_dim)

        # weight initialization
        fcn = [self.fc0, self.fc1, self.fc2, self.fc3]
        for f in fcn:
            nn.init.kaiming_uniform_(f.weight, a=0.01)

    def forward(self, x):
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class QRDQN():
    def __init__(self, env, device):
        self.s_dim = env.s_dim + 1

        # single_dim_mesh = torch.tensor([-3., -2., -1., -.9, -.5, -.2, 0., .2, .5, .9, 1., 2., 3.])  # M
        # single_dim_mesh = torch.tensor([-.9, -.5, -.2, .0, .2, .5, .9])  # M
        single_dim_mesh = torch.tensor([-30., -15., -10., -5., -3., -.9, -.5, -.2, -.1, -.05, -.02, -.01, 0.,
                                        .01, .02, .05, .1, .2, .5, .9, 3., 5., 10.])  # M
        n_grid = len(single_dim_mesh)
        self.env_a_dim = env.a_dim
        self.a_dim = n_grid ** self.env_a_dim  # M ** A
        self.a_mesh = torch.stack(torch.meshgrid([single_dim_mesh for _ in range(self.env_a_dim)]))  # (A, M, M, .., M)
        self.a_mesh_idx = torch.arange(self.a_dim).view(*[n_grid for _ in range(self.env_a_dim)])  # (A, M, M, .., M)

        self.prev_a_idx = None

        self.device = device

        self.replay_buffer = ReplayBuffer(env, device, buffer_size=BUFFER_SIZE, batch_size=MINIBATCH_SIZE)
        self.exp_noise = OU_Noise(1, RANDOM_SEED)
        self.epsilon = None  # initial value, if not updated before use, will raise an exception

        self.q_net = DeepNetwork(self.s_dim, self.a_dim * NUM_QUANT).to(device)  # (s) --> (a, N)
        self.target_q_net = DeepNetwork(self.s_dim, self.a_dim * NUM_QUANT).to(device)  # (s) --> (a, N)

        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.huber_loss = torch.nn.SmoothL1Loss(reduction='none')
        self.tau = ((2 * torch.arange(NUM_QUANT) + 1) / (2. * NUM_QUANT)).unsqueeze(0)
        self.loss = torch.tensor([0.], device=self.device)
        self.loss_sum = None
        self.loss_record = torch.tensor([0.], device=self.device)

        self.q_net_opt = torch.optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS, weight_decay=L2REG)

        K = torch.tensor([[67.], [33.]], requires_grad=False)
        self.pid = PID(env, device, K_val=K)

    def ctrl(self, epi, step, *single_expr):
        t, x, u, r, t2, x2, term, derivs = single_expr
        u_idx = self.prev_a_idx
        if u_idx is None: u_idx = torch.tensor([(self.a_dim - 1) / 2], dtype=torch.int64)

        single_expr = (t, x, u_idx, r, t2, x2, term, *derivs)
        self.replay_buffer.add(*single_expr)

        if epi < INITIAL_POLICY_INDEX:
            a_idx, a_val = self.initial_policy(step, t, x, u)
        else:
            a_idx, a_val = self.choose_action(epi, step, t, x)
            self.train(step)
        self.prev_a_idx = a_idx
        if term:
            np.savetxt(my_CSTR + '\\data\\loss_record.txt', self.loss_record.detach().cpu().numpy(), newline='\n')
        return a_val

    def choose_action(self, epi, step, t, x):
        tx = torch.cat([t,x] , 1)
        self.target_q_net.eval()
        with torch.no_grad():
            a_idx = self.get_value_distribution(self.target_q_net, tx).mean(2).min(1)[1]
        self.target_q_net.train()

        if step == 0:
            self.exp_schedule(epi)
            if self.loss_sum is not None:
                self.loss_record = torch.cat((self.loss_record, torch.tensor([self.loss_sum])))
            self.loss_sum = 0.

        explore_possibility = random.random()
        if explore_possibility <= self.epsilon and (epi + 1) % 50 > 0:
            a_idx = torch.randint(self.a_dim, [1, ])
        a_nom = self.action_idx2mesh(vec_idx=a_idx)

        return a_idx, a_nom

    def exp_schedule(self, epi):
        self.epsilon = EPSILON / (1 + EPI_DENOM * epi)

    def action_idx2mesh(self, vec_idx):
        mesh_idx = (self.a_mesh_idx == vec_idx).nonzero().squeeze(0)
        a_nom = torch.tensor([self.a_mesh[i, :][tuple(mesh_idx)] for i in range(self.env_a_dim)]).float().unsqueeze(
            0).to(self.device)
        return a_nom

    def train(self, step):
        t_batch, s_batch, a_batch, r_batch, t2_batch, s2_batch, term_batch, _, _, _, _ = self.replay_buffer.sample()
        ts_batch = torch.cat([t_batch, s_batch], 1)
        ts2_batch = torch.cat([t2_batch, s2_batch], 1)

        q_batch = self.get_value_distribution(self.q_net, ts_batch)[torch.arange(MINIBATCH_SIZE), a_batch]

        q2_distribution = self.get_value_distribution(self.target_q_net, ts2_batch)
        a_max_idx_batch = q2_distribution.mean(2).min(1)[1]
        q2_batch = q2_distribution[torch.arange(MINIBATCH_SIZE), a_max_idx_batch]
        q_target = r_batch + -(-1 + term_batch.float()) * q2_batch

        loss = self.quantile_huber_loss(q_batch, q_target)
        with torch.no_grad():
            self.loss_sum += loss

        self.q_net_opt.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.q_net_opt.step()
        self.loss = loss.detach_().unsqueeze(0)

        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(TAU * from_model.data + (1 - TAU) * to_model.data)

    def get_value_distribution(self, net, s):
        return net(s).view(-1, self.a_dim, NUM_QUANT)

    def quantile_huber_loss(self, q_batch, q_target_batch):
        qh_loss_batch = torch.tensor(0., device=self.device)
        for n, q in enumerate(q_batch):
            q_target = q_target_batch[n]
            error = q_target.unsqueeze(0) - q.unsqueeze(1)
            huber_loss = self.huber_loss(error, torch.zeros(error.shape, device=self.device))
            qh_loss = (huber_loss * (self.tau - (error < 0).float()).abs()).mean(1).sum(0)
            qh_loss_batch = qh_loss_batch + qh_loss
        return qh_loss_batch

    def initial_policy(self, step, t, x, u):
        u2 = self.pid.pid_ctrl(step, t, x)

        # get the closest value from action dimension
        distance_min = None
        for m in range(self.a_dim):
            a_point = self.action_idx2mesh(m)
            distance = torch.matmul(a_point - u2, (a_point - u2).T)
            if distance_min is None:
                u_min, distance_min, idx_min = a_point, distance, m
            elif distance_min >= distance:
                u_min, distance_min, idx_min = a_point, distance, m

        idx_min = torch.tensor([idx_min], device=self.device)
        return idx_min, u_min