import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .base_algorithm import Algorithm
from replay_buffer.replay_buffer import ReplayBuffer
from network.network import ActorMlp, CriticMLP
from utility.explorers import OUNoise


class DDPG(Algorithm):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.s_dim = self.config.s_dim
        self.a_dim = self.config.a_dim
        self.nT = self.config.nT

        # Hyperparameters
        self.num_hidden_nodes = self.config.num_hidden_nodes
        self.num_hidden_layers = self.config.num_hidden_layers
        hidden_dim_lst = [self.num_hidden_nodes for _ in range(self.num_hidden_layers)]

        self.crt_learning_rate = self.config.critic_lr
        self.act_learning_rate = self.config.actor_lr
        self.adam_eps = self.config.adam_eps
        self.l2_reg = self.config.l2_reg
        self.grad_clip_mag = self.config.grad_clip_mag
        self.tau = self.config.tau

        self.explorer = OUNoise(config)
        self.replay_buffer = ReplayBuffer(config)

        # Critic network
        self.critic = CriticMLP(self.s_dim+self.a_dim, 1, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic = CriticMLP(self.s_dim + self.a_dim, 1, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor network
        self.actor = ActorMlp(self.s_dim, self.a_dim, hidden_dim_lst, F.silu).to(self.device)
        self.target_actor = ActorMlp(self.s_dim, self.a_dim, hidden_dim_lst, F.silu).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.loss_lst = ['Critic loss', 'Actor loss']

    def ctrl(self, state):
        state = torch.from_numpy(state.T).float().to(self.device)
        action = self.actor(state, deterministic=True)
        action = self.explorer.sample(action.T.cpu().detach().numpy())
        action = np.clip(action, -1., 1.)

        return action

    def add_experience(self, *single_expr):
        state, action, reward, next_state, done = single_expr
        self.replay_buffer.add(*[state, action, reward, next_state, done])

    def train(self):
        assert len(self.replay_buffer) > 0

        # Replay buffer sample
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Network update
        critic_loss = self._critic_update(states, actions, rewards, next_states, dones)
        actor_loss = self._actor_update(states)
        self._target_net_update()

        loss = np.array([critic_loss, actor_loss])

        return loss

    def _critic_update(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions = self.target_actor(next_states, deterministic=True)
            next_q = self.critic(torch.cat([next_states, next_actions], dim=-1)).detach()
            target_q = rewards + next_q * (1 - dones)

        current_q = self.critic(torch.cat([states, actions], dim=-1))
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
        self.critic_optimizer.step()

        return critic_loss.detach().cpu().item()

    def _actor_update(self, states):
        a_pred_batch = self.actor(states, deterministic=True)
        actor_loss = self.target_critic(torch.cat([states, a_pred_batch], dim=-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_mag)
        self.actor_optimizer.step()

        return actor_loss.detach().cpu().item()

    def _target_net_update(self):
        for to_model, from_model in zip(self.target_critic.parameters(), self.critic.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        for to_model, from_model in zip(self.target_actor.parameters(), self.actor.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

    def save(self, path, file_name):
        torch.save(self.critic.state_dict(), os.path.join(path, file_name + '_critic.pt'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, file_name + '_critic_optimizer.pt'))

        torch.save(self.actor.state_dict(), os.path.join(path, file_name + '_actor.pt'))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, file_name + '_actor_optimizer.pt'))

    def load(self, path, file_name):
        self.critic.load_state_dict(torch.load(os.path.join(path, file_name + '_critic.pt')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic_optimizer.pt')))
        self.target_critic = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(path, file_name + '_actor.pt')))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_actor_optimizer.pt')))
        self.target_actor = copy.deepcopy(self.actor)
