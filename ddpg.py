import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from replay_buffer import ReplayBuffer


class DDPG(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim

        # Hyperparameters
        self.h_nodes = self.config.hyperparameters['hidden_nodes']
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.explore_epi_idx = self.config.hyperparameters['explore_epi_idx']
        self.buffer_size = self.config.hyperparameters['buffer_size']
        self.minibatch_size = self.config.hyperparameters['minibatch_size']
        self.crt_learning_rate = self.config.hyperparameters['critic_learning_rate']
        self.act_learning_rate = self.config.hyperparameters['actor_learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.tau = self.config.hyperparameters['tau']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.minibatch_size)

        # Critic network
        self.critic_net = self.approximator(self.s_dim + self.a_dim, 1, self.h_nodes).to(self.device)
        self.target_critic_net = self.approximator(self.s_dim + self.a_dim, 1, self.h_nodes).to(self.device)

        for to_model, from_model in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.critic_net_opt = optim.Adam(self.critic_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor network
        self.actor_net = self.approximator(self.s_dim, self.a_dim, self.h_nodes).to(self.device)
        self.target_actor_net = self.approximator(self.s_dim, self.a_dim, self.h_nodes).to(self.device)

        for to_model, from_model in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.actor_net_opt = optim.Adam(self.actor_net.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.loss_lst = ['Critic_loss', 'Actor_loss']

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

        self.actor_net.eval()
        with torch.no_grad():
            a = self.actor_net(s)
        self.actor_net.train()

        # Torch to Numpy
        a = a.T.cpu().detach().numpy()

        return a

    def add_experience(self, *single_expr):
        s, a, r, s2, is_term = single_expr
        self.replay_buffer.add(*[s, a, r, s2, is_term])

    def train(self):
        if len(self.replay_buffer) > 0:
            # Replay buffer sample
            s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample()

            # Network update
            critic_loss = self._critic_update(s_batch, a_batch, r_batch, s2_batch, term_batch)
            actor_loss = self._actor_update(s_batch)
            self._target_net_update()

            loss = np.array([critic_loss, actor_loss])
        else:
            loss = np.array([0., 0.])

        return loss

    def _critic_update(self, s_batch, a_batch, r_batch, s2_batch, term_batch):
        with torch.no_grad():
            a2_batch = self.target_actor_net(s2_batch)
            q_target_batch = r_batch + self.critic_net(torch.cat([s2_batch, a2_batch], dim=-1)).detach() * (1 - term_batch)

        q_batch = self.critic_net(torch.cat([s_batch, a_batch], dim=-1))
        critic_loss = F.mse_loss(q_batch, q_target_batch)

        self.critic_net_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.grad_clip_mag)
        self.critic_net_opt.step()

        return critic_loss.detach().cpu().item()

    def _actor_update(self, s_batch):
        a_pred_batch = self.actor_net(s_batch)
        actor_loss = self.target_critic_net(torch.cat([s_batch, a_pred_batch], dim=-1)).mean()

        self.actor_net_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.grad_clip_mag)
        self.actor_net_opt.step()

        return actor_loss.detach().cpu().item()

    def _target_net_update(self):
        for to_model, from_model in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        for to_model, from_model in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

