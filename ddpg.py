import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_create import NeuralNetworks
from replay_buffer import ReplayBuffer
from ou_noise import OU_Noise

class DDPG(object):
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
        self.crt_learning_rate = self.config.hyperparameters['critic_learning_rate']
        self.act_learning_rate = self.config.hyperparameters['actor_learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.eps = self.config.hyperparameters['eps_greedy']
        self.epi_denom = self.config.hyperparameters['eps_greedy_denom']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.tau = self.config.hyperparameters['tau']

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.minibatch_size)
        self.exp_noise = OU_Noise(self.a_dim)
        self.initial_ctrl = InitialControl(self.config)

        # Critic (+target) net
        self.q_net = NeuralNetworks(self.s_dim + self.a_dim, 1).to(self.device)
        self.target_q_net = NeuralNetworks(self.s_dim + self.a_dim, 1).to(self.device)

        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.q_net_opt = torch.optim.Adam(self.q_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor (+target) net
        self.mu_net = NeuralNetworks(self.s_dim, self.a_dim).to(self.device)
        self.target_mu_net = NeuralNetworks(self.s_dim, self.a_dim).to(self.device)

        for to_model, from_model in zip(self.target_mu_net.parameters(), self.mu_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.mu_net_opt = torch.optim.Adam(self.mu_net.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

    def ctrl(self, epi, step, x, u):
        a_exp = self.exp_schedule(epi, step)
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.controller(epi, step, x, u)
        elif self.init_ctrl_idx <= epi < self.explore_epi_idx:
            a_nom = self.choose_action(epi, step, x, u)
        else:
            a_nom = self.choose_action(epi, step, x, u)

        a_nom.detach()
        a_val = a_nom + torch.tensor(a_exp, dtype=torch.float, device=self.device)
        if epi>= 1: self.nn_train()
        return a_val

    def add_experience(self, *single_expr):
        x, u, r, x2, term = single_expr
        self.replay_buffer.add(*[x, u, r, x2, term])

    def choose_action(self, epi, step, x, u):
        # Option: target_actor_net OR actor_net?
        self.mu_net.eval()
        with torch.no_grad():
            a_nom = self.mu_net(x)
        self.mu_net.train()
        return a_nom

    def exp_schedule(self, epi, step):
        noise = self.exp_noise.sample() / 10.
        if step == 0: self.epsilon = self.eps / (1. + (epi / self.epi_denom))
        a_exp = noise * self.epsilon
        return a_exp

    def nn_train(self):
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

class InitialControl(object):
    def __init__(self, config):
        from pid import PID
        self.pid = PID(config.env, config.device)

    def controller(self, epi, step, x, u):
        return self.pid.ctrl(epi, step, x, u)
