import torch
import torch.nn.functional as F
import numpy as np

from replay_buffer import ReplayBuffer

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

        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.tau = self.config.hyperparameters['tau']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.buffer_size)

        # Critic (+target) net
        self.critic_net = self.approximator(self.s_dim, 1, self.crt_h_nodes).to(self.device)  # s --> 1
        self.target_critic_net = self.approximator(self.s_dim, 1, self.crt_h_nodes).to(self.device) # s --> 1

        for to_model, from_model in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.critic_net_opt = torch.optim.Adam(self.critic_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor (+target) net
        self.actor_net = self.approximator(self.s_dim, self.a_dim, self.act_h_nodes).to(self.device)  # s --> a
        self.target_actor_net = self.approximator(self.s_dim, self.a_dim, self.act_h_nodes).to(self.device)  # s --> a

        for to_model, from_model in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.actor_net_opt = torch.optim.Adam(self.actor_net.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Costate (+target) net
        self.costate_net = self.approximator(self.s_dim, self.s_dim, self.cos_h_nodes).to(self.device)  # s --> s
        self.target_costate_net = self.approximator(self.s_dim, self.s_dim, self.cos_h_nodes).to(self.device)  # s --> s

        for to_model, from_model in zip(self.target_costate_net.parameters(), self.costate_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.costate_net_opt = torch.optim.Adam(self.costate_net.parameters(), lr=self.cst_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

    def ctrl(self, epi, step, x, u):
        if epi < self.init_ctrl_idx:
            u_nom = self.initial_ctrl.ctrl(epi, step, x, u)
            u_val = self.explorer.sample(epi, step, u_nom)
        elif self.init_ctrl_idx <= epi < self.explore_epi_idx:
            u_nom = self.choose_action(epi, step, x, u)
            u_val = self.explorer.sample(epi, step, u_nom)
        else:
            u_val = self.choose_action(epi, step, x, u)

        return u_val

    def choose_action(self, epi, step, x, u):
        # Numpy to torch
        x = torch.from_numpy(x.T).float().to(self.device)  # (B, 1)

        # Option: target_actor_net OR actor_net?
        self.target_actor_net.eval()
        with torch.no_grad():
            u = self.target_actor_net(x)
        self.target_actor_net.train()

        # Torch to Numpy
        u = u.T.detach().numpy()
        return u

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

        if len(self.replay_buffer) > 0:
            # Replay buffer sample
            x_batch, u_batch, r_batch, x2_batch, term_batch, \
            dfdx_batch, dfdu_batch, dcdx_batch, d2cdu2inv_batch = self.replay_buffer.sample()

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

            loss = q_loss + l_loss + a_loss
            loss = loss.detach().numpy().item()
        else:
            loss = 0.

        return loss