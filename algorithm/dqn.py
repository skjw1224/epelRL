import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .base_algorithm import Algorithm
from network.nn import CriticMLP
from utility.replay_buffer import ReplayBuffer
from utility.explorers import EpsilonGreedy


class DQN(Algorithm):
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

        self.gamma = self.config.gamma
        self.critic_lr = self.config.critic_lr
        self.adam_eps = self.config.adam_eps
        self.l2_reg = self.config.l2_reg
        self.grad_clip_mag = self.config.grad_clip_mag
        self.tau = self.config.tau
        self.single_dim_mesh = self.config.single_dim_mesh

        self.explorer = EpsilonGreedy(config)
        self.replay_buffer = ReplayBuffer(config)

        # Action mesh
        self._generate_action_mesh()

        # Critic network (Q network)
        self.critic = CriticMLP(self.s_dim, self.a_mesh_dim, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic = CriticMLP(self.s_dim, self.a_mesh_dim, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.loss_lst = ['Critic loss']

    def _generate_action_mesh(self):
        # Generate action mesh and mesh index for discrete action space
        num_grid = len(self.single_dim_mesh)
        single_dim_mesh = np.array(self.single_dim_mesh)

        self.a_mesh_dim = num_grid ** self.a_dim
        self.a_mesh_idx = np.arange(self.a_mesh_dim).reshape(*[num_grid for _ in range(self.a_dim)])  # (M, M, .., M)
        self.a_mesh = np.stack(np.meshgrid(*[single_dim_mesh for _ in range(self.a_dim)]))  # (A, M, M, ..., M)

    def ctrl(self, state):
        with torch.no_grad():
            state = torch.tensor(state.T, dtype=torch.float32, device=self.device)
            q_values = self.critic(state)

        _, action_idx = torch.min(q_values, dim=1, keepdim=True)
        action_idx = action_idx.cpu().numpy()
        action_idx = self.explorer.sample(action_idx)
        action = self._idx2action(action_idx)

        return action

    def _idx2action(self, idx):
        # Get action values from indexes
        mesh_idx = np.where(self.a_mesh_idx == idx)
        action = np.array([self.a_mesh[i][mesh_idx] for i in range(self.a_dim)])

        return action

    def add_experience(self, *single_expr):
        state, action, reward, next_state, done = single_expr
        action_idx = self._action2idx(action)

        self.replay_buffer.add(*[state, action_idx, reward, next_state, done])

    def _action2idx(self, action):
        # Get indexes from action values
        action2idx_lst = [self.a_mesh[i] == action[i] for i in range(self.a_dim)]
        idx_lst = action2idx_lst[0]
        for i in range(1, len(action2idx_lst)):
            idx_lst = idx_lst & action2idx_lst[i]
        idx = np.where(idx_lst)
        mesh_idx = self.a_mesh_idx[idx].reshape(-1, 1)

        return mesh_idx

    def train(self):
        # Replay buffer sample
        states, action_indices, rewards, next_states, dones = self.replay_buffer.sample()

        # Compute the next Q-values using the target network
        with torch.no_grad():
            next_q = self.target_critic(next_states).detach()
            next_q, _ = torch.min(next_q, dim=1, keepdim=True)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Get current Q-values estimates
        current_q = self.critic(states)
        current_q = torch.gather(current_q, dim=1, index=action_indices.long())
        
        # Get critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Optimize the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
        self.critic_optimizer.step()

        # Soft update target network
        for to_model, from_model in zip(self.target_critic.parameters(), self.critic.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        loss = np.array([critic_loss.cpu().detach().numpy().item()])

        return loss

    def save(self, path, file_name):
        torch.save(self.critic.state_dict(), os.path.join(path, file_name + '_critic.pt'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, file_name + '_critic_optimizer.pt'))

    def load(self, path, file_name):
        self.critic.load_state_dict(torch.load(os.path.join(path, file_name + '_critic.pt')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic_optimizer.pt')))
        self.target_critic = copy.deepcopy(self.critic)
