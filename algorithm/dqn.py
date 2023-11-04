import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .base_algorithm import Algorithm
from network.network import CriticMLP
from replay_buffer.replay_buffer import ReplayBuffer
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

        self.critic_lr = self.config.critic_lr
        self.adam_eps = self.config.adam_eps
        self.l2_reg = self.config.l2_reg
        self.grad_clip_mag = self.config.grad_clip_mag
        self.tau = self.config.tau
        self.single_dim_mesh = self.config.single_dim_mesh

        self.explorer = EpsilonGreedy(config)
        self.replay_buffer = ReplayBuffer(config)

        # Action mesh
        self.a_mesh_dim, self.a_mesh, self.a_mesh_idx = self._generate_action_mesh()

        # Critic network (Q network)
        self.critic = CriticMLP(self.s_dim, self.a_mesh_dim, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic = CriticMLP(self.s_dim, self.a_mesh_dim, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.loss_lst = ['Critic loss']

    def ctrl(self, state):
        state = torch.from_numpy(state.T).float().to(self.device)
        action_idx = self.critic(state).min(-1)[1].unsqueeze(1)
        action_idx = action_idx.cpu().detach().numpy()
        action_idx = self.explorer.sample(action_idx)

        action = self.idx2action(action_idx)

        return action

    def _generate_action_mesh(self):
        num_grid = len(self.single_dim_mesh)
        a_mesh_dim = num_grid ** self.a_dim
        single_dim_mesh = np.array(self.single_dim_mesh)
        a_mesh = np.stack(np.meshgrid(*[single_dim_mesh for _ in range(self.a_dim)]))  # (A, M, M, ..., M)
        a_mesh_idx = np.arange(a_mesh_dim).reshape(*[num_grid for _ in range(self.a_dim)])  # (M, M, .., M)

        return a_mesh_dim, a_mesh, a_mesh_idx

    def idx2action(self, idx):
        mesh_idx = np.where(self.a_mesh_idx == idx)
        action = np.array([self.a_mesh[i][mesh_idx] for i in range(self.a_dim)])

        return action

    def action2idx(self, action):
        action2idx_lst = [self.a_mesh[i] == action[i] for i in range(self.a_dim)]
        idx_lst = action2idx_lst[0]
        for i in range(1, len(action2idx_lst)):
            idx_lst = idx_lst & action2idx_lst[i]
        idx = np.where(idx_lst)
        mesh_idx = self.a_mesh_idx[idx].reshape(-1, 1)

        return mesh_idx

    def add_experience(self, *single_expr):
        s, a, r, s2, is_term = single_expr
        a_idx = self.action2idx(a)

        self.replay_buffer.add(*[s, a_idx, r, s2, is_term])

    def train(self):
        assert (len(self.replay_buffer) > 0)

        # Replay buffer sample
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Network update
        with torch.no_grad():
            next_q = self.target_critic(next_states).detach().min(-1)[0].unsqueeze(1) * (1 - dones)
            target_q = rewards + next_q

        current_q = self.critic(states).gather(1, actions.long())
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
        self.critic_optimizer.step()

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
