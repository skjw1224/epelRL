import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .base_algorithm import Algorithm
from replay_buffer.replay_buffer import ReplayBuffer
from network.network import ActorMlp, CriticMLP


class SAC(Algorithm):
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.s_dim = self.config.s_dim
        self.a_dim = self.config.a_dim
        self.nT = self.config.nT

        # Hyperparameters
        self.num_hidden_nodes = self.config.num_hidden_nodes
        self.num_hidden_layers = self.config.num_hidden_layers
        hidden_dim_lst = [self.num_hidden_nodes for _ in range(self.num_hidden_layers)]

        self.critic_lr = self.config.critic_lr
        self.actor_lr = self.config.actor_lr
        self.adam_eps = self.config.adam_eps
        self.l2_reg = self.config.l2_reg
        self.grad_clip_mag = self.config.grad_clip_mag
        self.tau = self.config.tau

        self.replay_buffer = ReplayBuffer(config)

        # Critic networks
        self.critic1 = CriticMLP(self.s_dim+self.a_dim, 1, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic1 = CriticMLP(self.s_dim+self.a_dim, 1, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.critic2 = CriticMLP(self.s_dim+self.a_dim, 1, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic2 = CriticMLP(self.s_dim+self.a_dim, 1, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor net
        self.actor = ActorMlp(self.s_dim, self.a_dim, hidden_dim_lst, F.silu).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Temperature learning
        self.automatic_temp_tuning = self.config.automatic_temp_tuning
        if self.automatic_temp_tuning:
            self.target_entropy = - self.a_dim
            self.log_temp = torch.zeros(1, requires_grad=True, device=self.device)
            self.temp = self.log_temp.exp()
            self.temp_optimizer = optim.Adam([self.log_temp], lr=self.actor_lr, eps=self.adam_eps, weight_decay=self.l2_reg) # 혹은 따로 지정
        else:
            self.temp = self.config.temperature

        self.loss_lst = ['Critic1 loss', 'Critic2 loss', 'Actor loss', 'Temp loss']

    def ctrl(self, state):
        state = torch.from_numpy(state.T).float().to(self.device)
        action, _ = self.actor(state, deterministic=False, reparam_trick=True, return_log_prob=False)
        action = np.clip(action.T.detach().cpu().numpy(), -1., 1.)

        return action

    def add_experience(self, *single_expr):
        state, action, reward, next_state, done = single_expr
        self.replay_buffer.add(*[state, action, reward, next_state, done])

    def train(self):
        # Replay buffer sample
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Network update
        critic1_loss, critic2_loss = self._critic_update(states, actions, rewards, next_states, dones)
        actor_loss, temp_loss = self._actor_update(states)
        self._soft_update()

        loss = np.array([critic1_loss, critic2_loss, actor_loss, temp_loss])

        return loss

    def _critic_update(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, log_probs = self.actor(next_states, deterministic=False, reparam_trick=True, return_log_prob=True)
            next_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=-1)).detach()
            next_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=-1)).detach()
            next_q = torch.max(next_q1, next_q2) - self.temp * log_probs.sum(1, keepdim=True)
            target_q = rewards + next_q * (1 - dones)

        current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=-1))
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip_mag)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip_mag)
        self.critic2_optimizer.step()

        return critic1_loss.detach().cpu().item(), critic2_loss.detach().cpu().item()

    def _actor_update(self, states):
        actions, log_probs = self.actor(states, deterministic=False, reparam_trick=True, return_log_prob=True)
        current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=-1))
        actor_loss = (torch.max(current_q1, current_q2) - (self.temp * log_probs.sum(1, keepdim=True))).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_mag)
        self.actor_optimizer.step()

        if self.automatic_temp_tuning:
            temp_loss = - (self.log_temp * (log_probs + self.target_entropy).detach()).mean()
            self.temp_optimizer.zero_grad()
            temp_loss.backward()
            self.temp_optimizer.step()
            self.temp = self.log_temp.exp()
            temp_loss = temp_loss.detach().cpu().item()
        else:
            temp_loss = 0

        return actor_loss.detach().cpu().item(), temp_loss

    def _soft_update(self):
        for to_model, from_model in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        for to_model, from_model in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

    def save(self, path, file_name):
        torch.save(self.critic1.state_dict(), os.path.join(path, file_name + '_critic.pt'))
        torch.save(self.critic1_optimizer.state_dict(), os.path.join(path, file_name + '_critic_optimizer.pt'))

        torch.save(self.critic2.state_dict(), os.path.join(path, file_name + '_critic.pt'))
        torch.save(self.critic2_optimizer.state_dict(), os.path.join(path, file_name + '_critic_optimizer.pt'))

        torch.save(self.actor.state_dict(), os.path.join(path, file_name + '_actor.pt'))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, file_name + '_actor_optimizer.pt'))

        if self.automatic_temp_tuning:
            torch.save(self.log_temp, os.path.join(path, file_name + '_temp.pt'))
            torch.save(self.temp_optimizer.state_dict(), os.path.join(path, file_name + '_temp_optimizer.pt'))

    def load(self, path, file_name):
        self.critic1.load_state_dict(torch.load(os.path.join(path, file_name + '_critic.pt')))
        self.critic1_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic_optimizer.pt')))
        self.target_critic1 = copy.deepcopy(self.critic1)

        self.critic2.load_state_dict(torch.load(os.path.join(path, file_name + '_critic.pt')))
        self.critic2_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic_optimizer.pt')))
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(os.path.join(path, file_name + '_actor.pt')))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_actor_optimizer.pt')))

        if self.automatic_temp_tuning:
            self.log_temp = torch.load(os.path.join(path, file_name + '_temp.pt'))
            self.temp_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_temp_optimizer.pt')))
