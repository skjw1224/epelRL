import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from algorithm import Algorithm
from replay_buffer import ReplayBuffer


class A2C(Algorithm):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        # Hyperparameters
        self.n_step_TD = self.config.hyperparameters['n_step_TD']
        self.crt_h_nodes = self.config.hyperparameters['hidden_nodes']
        self.act_h_nodes = self.config.hyperparameters['hidden_nodes']
        self.crt_learning_rate = self.config.hyperparameters['critic_learning_rate']
        self.act_learning_rate = self.config.hyperparameters['actor_learning_rate']
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.explore_epi_idx = self.config.hyperparameters['explore_epi_idx']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT, batch_size=self.nT)

        # Critic network
        self.critic = self.approximator(self.s_dim, 1, self.crt_h_nodes).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor network
        self.actor = self.approximator(self.s_dim, 2 * self.a_dim, self.act_h_nodes).to(self.device)
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.loss_lst = ['Critic loss', 'Actor loss']

    def ctrl(self, epi, step, s, a):
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.ctrl(epi, step, s, a)
            a_val = self.explorer.sample(epi, step, a_nom)
        else:
            a_val = self._choose_action(s)

        a_val = np.clip(a_val, -1., 1.)

        return a_val

    def add_experience(self, *single_expr):
        s, a, r, s2, is_term = single_expr
        self.replay_buffer.add(*[s, a, r, s2, is_term])

    def _choose_action(self, s):
        # numpy to torch
        s = torch.from_numpy(s.T).float().to(self.device)

        self.actor.eval()
        with torch.no_grad():
            a_pred = self.actor(s)
        self.actor.train()

        mean, log_std = a_pred[:, :self.a_dim], a_pred[:, self.a_dim:]
        std = torch.exp(log_std)
        a_distribution = Normal(mean, std)
        a = a_distribution.sample()
        a = torch.tanh(a)

        # torch to numpy
        a = a.T.cpu().detach().numpy()

        return a

    def _get_log_prob(self, s_batch, a_batch):
        a_pred = self.actor(s_batch)
        mean, log_std = a_pred[:, :self.a_dim], a_pred[:, self.a_dim:]
        std = torch.exp(log_std)
        distribution = Normal(mean, std)
        log_prob = distribution.log_prob(a_batch)

        return log_prob

    def train(self):
        s_traj, a_traj, r_traj, s2_traj, term_traj = self.replay_buffer.sample_sequence()
        log_prob_traj = self._get_log_prob(s_traj, a_traj)

        v_target_traj = []

        if term_traj[-1]:  # When Final value of sequence is terminal sample
            v_target_traj.append(r_traj[-1])  # Append terminal cost
        else:  # When Final value of sequence is path sample
            v_target_traj.append(self.critic(s2_traj[-1]))  # Append n-step bootstrapped q-value

        for i in range(len(s_traj)):
            v_target_traj.append(r_traj[-i-1] + v_target_traj[-1])

        v_target_traj.reverse()
        v_target_traj = torch.stack(v_target_traj[:-1])
        v_target_traj.detach()

        v_traj = self.critic(s_traj)
        advantage_traj = v_target_traj - v_traj

        critic_loss = F.mse_loss(v_target_traj, v_traj)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
        self.critic_optimizer.step()

        actor_loss = (log_prob_traj * advantage_traj.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_mag)
        self.actor_optimizer.step()

        critic_loss = critic_loss.detach().cpu().item()
        actor_loss = actor_loss.detach().cpu().item()
        loss = np.array([critic_loss, actor_loss])

        self.replay_buffer.clear()

        return loss

    def save(self, path, file_name):
        torch.save(self.critic.state_dict(), os.path.join(path, file_name + '_critic.pt'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, file_name + '_critic_optimizer.pt'))

        torch.save(self.actor.state_dict(), os.path.join(path, file_name + '_actor.pt'))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, file_name + '_actor_optimizer.pt'))

    def load(self, path, file_name):
        self.critic.load_state_dict(torch.load(os.path.join(path, file_name + '_critic.pt')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic_optimizer.pt')))

        self.actor.load_state_dict(torch.load(os.path.join(path, file_name + '_actor.pt')))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_actor_optimizer.pt')))
