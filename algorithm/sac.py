import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .base_algorithm import Algorithm
from network.nn import ActorMlp, CriticMLP
from utility.buffer import ReplayBuffer


class SAC(Algorithm):
    def __init__(self, config):
        self.config = config
        self.device = self.config['device']
        self.s_dim = self.config['s_dim']
        self.a_dim = self.config['a_dim']
        self.nT = self.config['nT']

        # Hyperparameters
        self.num_hidden_nodes = self.config['num_hidden_nodes']
        self.num_hidden_layers = self.config['num_hidden_layers']
        hidden_dim_lst = [self.num_hidden_nodes for _ in range(self.num_hidden_layers)]

        self.gamma = self.config['gamma']
        self.critic_lr = self.config['critic_lr']
        self.actor_lr = self.config['actor_lr']
        self.adam_eps = self.config['adam_eps']
        self.l2_reg = self.config['l2_reg']
        self.grad_clip_mag = self.config['grad_clip_mag']
        self.tau = self.config['tau']

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

        # Actor network
        self.actor = ActorMlp(self.s_dim, self.a_dim, hidden_dim_lst, F.silu).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Temperature learning
        self.automatic_temp_tuning = self.config['automatic_temp_tuning']
        if self.automatic_temp_tuning:
            self.target_entropy = - self.a_dim
            self.log_temp = torch.zeros(1, requires_grad=True, device=self.device)
            self.temp = self.log_temp.exp()
            self.temp_optimizer = optim.Adam([self.log_temp], lr=self.actor_lr, eps=self.adam_eps, weight_decay=self.l2_reg)
        else:
            self.temp = self.config['temperature']

        self.loss_lst = ['Critic1 loss', 'Critic2 loss', 'Actor loss', 'Temp loss']

    def ctrl(self, state):
        with torch.no_grad():
            state = torch.tensor(state.T, dtype=torch.float32, device=self.device)
            action, _ = self.actor(state, deterministic=False, reparam_trick=True, return_log_prob=False)

        action = np.clip(action.T.cpu().numpy(), -1., 1.)

        return action

    def add_experience(self, experience):
        self.replay_buffer.add(experience)

    def warm_up_train(self):
        pass

    def train(self):
        # Replay buffer sample
        sample = self.replay_buffer.sample()
        states = sample['states']
        actions = sample['actions']
        rewards = sample['rewards']
        next_states = sample['next_states']
        dones = sample['dones']

        # Compute the next Q values using the target values
        with torch.no_grad():
            next_actions, log_probs = self.actor(next_states, deterministic=False, reparam_trick=True, return_log_prob=True)
            next_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=-1))
            next_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=-1))
            next_q = torch.max(next_q1, next_q2) - self.temp * log_probs.sum(dim=1, keepdim=True)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute critic loss & Optimize the critic networks
        current_q1 = self.critic1(torch.cat([states, actions], dim=-1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=-1))
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_clip_mag)
        self.critic1_optimizer.step()
        critic1_loss = critic1_loss.detach().cpu().item()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_clip_mag)
        self.critic2_optimizer.step()
        critic2_loss = critic2_loss.detach().cpu().item()

        # Compute actor loss & Optimize the actor network
        actor_actions, actor_log_probs = self.actor(states, deterministic=False, reparam_trick=True, return_log_prob=True)
        q1 = self.critic1(torch.cat([states, actor_actions], dim=-1))
        q2 = self.critic2(torch.cat([states, actor_actions], dim=-1))
        actor_loss = (torch.max(q1, q2) + (self.temp.detach() * actor_log_probs.sum(dim=1, keepdim=True))).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_mag)
        self.actor_optimizer.step()
        actor_loss = actor_loss.detach().cpu().item()

        # Compute temperature loss & Optimize the temperature
        if self.automatic_temp_tuning:
            temp_loss = - (self.log_temp * (actor_log_probs + self.target_entropy).detach()).mean()
            self.temp_optimizer.zero_grad()
            temp_loss.backward()
            self.temp_optimizer.step()
            self.temp = self.log_temp.exp()
            temp_loss = temp_loss.detach().cpu().item()
        else:
            temp_loss = 0

        # Soft update the target networks
        for to_model, from_model in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        for to_model, from_model in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        loss = np.array([critic1_loss, critic2_loss, actor_loss, temp_loss])

        return loss

    def save(self, path, file_name):
        torch.save(self.critic1.state_dict(), os.path.join(path, file_name + '_critic1.pt'))
        torch.save(self.critic1_optimizer.state_dict(), os.path.join(path, file_name + '_critic1_optimizer.pt'))

        torch.save(self.critic2.state_dict(), os.path.join(path, file_name + '_critic2.pt'))
        torch.save(self.critic2_optimizer.state_dict(), os.path.join(path, file_name + '_critic2_optimizer.pt'))

        torch.save(self.actor.state_dict(), os.path.join(path, file_name + '_actor.pt'))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, file_name + '_actor_optimizer.pt'))

        if self.automatic_temp_tuning:
            torch.save(self.log_temp, os.path.join(path, file_name + '_temp.pt'))
            torch.save(self.temp_optimizer.state_dict(), os.path.join(path, file_name + '_temp_optimizer.pt'))

    def load(self, path, file_name):
        self.critic1.load_state_dict(torch.load(os.path.join(path, file_name + '_critic1.pt')))
        self.critic1_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic1_optimizer.pt')))
        self.target_critic1 = copy.deepcopy(self.critic1)

        self.critic2.load_state_dict(torch.load(os.path.join(path, file_name + '_critic2.pt')))
        self.critic2_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic2_optimizer.pt')))
        self.target_critic2 = copy.deepcopy(self.critic2)

        self.actor.load_state_dict(torch.load(os.path.join(path, file_name + '_actor.pt')))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_actor_optimizer.pt')))

        if self.automatic_temp_tuning:
            self.log_temp = torch.load(os.path.join(path, file_name + '_temp.pt'))
            self.temp_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_temp_optimizer.pt')))
