import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .algorithm import Algorithm
from replay_buffer.replay_buffer import ReplayBuffer


class GDHP(Algorithm):
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

        # Critic network
        self.critic = self.approximator(self.s_dim, 1, self.crt_h_nodes).to(self.device)  # s --> 1
        self.target_critic = self.approximator(self.s_dim, 1, self.crt_h_nodes).to(self.device)  # s --> 1

        for to_model, from_model in zip(self.target_critic.parameters(), self.critic.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor network
        self.actor = self.approximator(self.s_dim, self.a_dim, self.act_h_nodes).to(self.device)  # s --> a
        self.target_actor = self.approximator(self.s_dim, self.a_dim, self.act_h_nodes).to(self.device)  # s --> a

        for to_model, from_model in zip(self.target_actor.parameters(), self.actor.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Costate network
        self.costate = self.approximator(self.s_dim, self.s_dim, self.cos_h_nodes).to(self.device)  # s --> s
        self.target_costate = self.approximator(self.s_dim, self.s_dim, self.cos_h_nodes).to(self.device)  # s --> s

        for to_model, from_model in zip(self.target_costate.parameters(), self.costate.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.costate_optimizer = optim.Adam(self.costate.parameters(), lr=self.cst_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.loss_lst = ['Critic loss', 'Costate loss', 'Actor loss']

    def ctrl(self, epi, step, s, a):
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.ctrl(epi, step, s, a)
            a_val = self.explorer.sample(epi, step, a_nom)
        else:
            a_nom = self._choose_action(s)
            a_val = self.explorer.sample(epi, step, a_nom)

        a_val = np.clip(a_val, -1., 1.)

        return a_val

    def _choose_action(self, s):
        # Numpy to torch
        s = torch.from_numpy(s.T).float().to(self.device)  # (B, 1)

        # Option: target_actor OR actor?
        self.target_actor.eval()
        with torch.no_grad():
            a = self.target_actor(s)
        self.target_actor.train()

        # Torch to Numpy
        a = a.T.detach().cpu().numpy()

        return a

    def add_experience(self, *single_expr):
        s, a, r, s2, is_term, derivs = single_expr
        self.replay_buffer.add(*[s, a, r, s2, is_term, *derivs])

    def train(self):
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
            s_batch, a_batch, r_batch, s2_batch, term_batch, \
            dfdx_batch, dfdu_batch, dcdx_batch, d2cdu2inv_batch = self.replay_buffer.sample()

            # Critic Train
            q_batch = self.critic(s_batch)
            q2_batch = self.target_critic(s2_batch).detach() * (1 - term_batch)
            q_target_batch = r_batch + q2_batch

            critic_loss = F.mse_loss(q_batch, q_target_batch)

            nn_update_one_step(self.critic, self.target_critic, self.critic_optimizer, critic_loss)

            # Costate Train
            l_batch = self.costate(s_batch)
            l2_batch = self.target_costate(s2_batch).detach() * (1 - term_batch)
            l_target_batch = (dcdx_batch.permute(0, 2, 1) + l2_batch.unsqueeze(1) @ dfdx_batch).squeeze(1) # (B, S)

            costate_loss = F.mse_loss(l_batch, l_target_batch)

            nn_update_one_step(self.costate, self.target_costate, self.costate_optimizer, costate_loss)

            # Actor Train
            a_batch = self.actor(s_batch)
            a_target_batch = torch.clamp((-0.5 * l2_batch.unsqueeze(1) @ dfdu_batch @ d2cdu2inv_batch), -1., 1.).detach().squeeze(1)

            actor_loss = F.mse_loss(a_batch, a_target_batch)

            nn_update_one_step(self.actor, self.target_actor, self.actor_optimizer, actor_loss)

            critic_loss = critic_loss.detach().cpu().item()
            costate_loss = costate_loss.detach().cpu().item()
            actor_loss = actor_loss.detach().cpu().item()
            loss = np.array([critic_loss, costate_loss, actor_loss])
        else:
            loss = np.array([0., 0., 0.])

        return loss

    def save(self, path, file_name):
        torch.save(self.critic.state_dict(), os.path.join(path, file_name + '_critic.pt'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, file_name + '_critic_optimizer.pt'))

        torch.save(self.actor.state_dict(), os.path.join(path, file_name + '_actor.pt'))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, file_name + '_actor_optimizer.pt'))

        torch.save(self.costate.state_dict(), os.path.join(path, file_name + '_costate.pt'))
        torch.save(self.costate_optimizer.state_dict(), os.path.join(path, file_name + '_costate_optimizer.pt'))

    def load(self, path, file_name):
        self.critic.load_state_dict(torch.load(os.path.join(path, file_name + '_critic.pt')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic_optimizer.pt')))
        self.target_critic = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(path, file_name + '_actor.pt')))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_actor_optimizer.pt')))
        self.target_actor = copy.deepcopy(self.actor)

        self.costate.load_state_dict(torch.load(os.path.join(path, file_name + '_costate.pt')))
        self.costate_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_costate_optimizer.pt')))
        self.target_costate = copy.deepcopy(self.costate)
        