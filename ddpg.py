import torch
import torch.nn as nn
import torch.nn.functional as F

from replay_buffer import ReplayBuffer
from ou_noise import OU_Noise



TAU = 0.05
EPSILON = 0.1
EPI_DENOM = 1.

CRT_LEARNING_RATE = 0.01
ACT_LEARNING_RATE = 0.001
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

    def forward(self, x):
        x = F.leaky_relu(self.bn0(self.fc0(x)))
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.fc3(x))
        return x


class DDPG(object):
    def __init__(self, env, device, replay_buffer):
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim

        self.device = device

        self.replay_buffer = replay_buffer
        self.exp_noise = OU_Noise(self.a_dim)
        self.initial_ctrl = InitialControl(env, device)

        # Critic (+target) net
        self.q_net = Network(self.s_dim + self.a_dim, 1).to(device)
        self.target_q_net = Network(self.s_dim + self.a_dim, 1).to(device)

        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.q_net_opt = torch.optim.Adam(self.q_net.parameters(), lr=CRT_LEARNING_RATE, eps=ADAM_EPS, weight_decay=L2REG)

        # Actor (+target) net
        self.mu_net = Network(self.s_dim, self.a_dim).to(device)
        self.target_mu_net = Network(self.s_dim, self.a_dim).to(device)

        for to_model, from_model in zip(self.target_mu_net.parameters(), self.mu_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.mu_net_opt = torch.optim.Adam(self.mu_net.parameters(), lr=ACT_LEARNING_RATE, eps=ADAM_EPS, weight_decay=L2REG)

    def ctrl(self, epi, step, x, u):
        a_exp = self.exp_schedule(epi, step)
        if epi < INITIAL_POLICY_INDEX:
            a_nom = self.initial_ctrl.controller(epi, step, x, u)
        elif INITIAL_POLICY_INDEX <= epi < AC_PE_TRAINING_INDEX:
            a_nom = self.choose_action(x)
        else:
            a_nom = self.choose_action(x)

        a_nom.detach()
        a_val = a_nom + torch.tensor(a_exp, dtype=torch.float, device=self.device)
        if epi>= 1: self.nn_train()
        return a_val

    def add_experience(self, *single_expr):
        x, u, r, x2, term = single_expr
        self.replay_buffer.add(*[x, u, r, x2, term])

    def choose_action(self, s):
        # Option: target_actor_net OR actor_net?
        self.mu_net.eval()
        with torch.no_grad():
            a_nom = self.mu_net(s)
        self.mu_net.train()
        return a_nom

    def exp_schedule(self, epi, step):
        noise = self.exp_noise.sample() / 10.
        if step == 0: self.epsilon = EPSILON / (1. + (epi / EPI_DENOM))
        a_exp = noise * self.epsilon
        return a_exp

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
        s_batch, a_batch, r_batch, s2_batch, term_batch, _, _ ,_, _ = self.replay_buffer.sample()

        # Critic Train
        q_batch = self.q_net(torch.cat([s_batch, a_batch], dim=-1))
        a2_batch = self.target_mu_net(s2_batch)

        q_target_batch = r_batch + self.q_net(torch.cat([s2_batch, a2_batch], dim=-1)) * (~term_batch)
        q_loss = F.mse_loss(q_batch, q_target_batch)

        nn_update_one_step(self.q_net, self.target_q_net, self.q_net_opt, q_loss)

        # Actor Train
        a_pred_batch = self.mu_net(s_batch)
        a_loss = self.target_q_net(torch.cat([s_batch, a_pred_batch], dim=-1)).mean()
        nn_update_one_step(self.mu_net, self.target_mu_net, self.mu_net_opt, a_loss)

from pid import PID
class InitialControl(object):
    def __init__(self, env, device):
        self.pid = PID(env, device)

    def controller(self, epi, step, x, u):
        return self.pid.ctrl(epi, step, x, u)