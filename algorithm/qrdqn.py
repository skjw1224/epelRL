import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .dqn import DQN
from .base_algorithm import Algorithm
from network.nn import CriticMLP
from utility.buffer import ReplayBuffer
from utility.explorers import EpsilonGreedy


class QRDQN(Algorithm):
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
        self.adam_eps = self.config['adam_eps']
        self.l2_reg = self.config['l2_reg']
        self.grad_clip_mag = self.config['grad_clip_mag']
        self.tau = self.config['tau']

        self.n_quantiles = self.config['n_quantiles']
        self.max_n_action_grid = self.config['max_n_action_grid']

        self.quantile_taus = ((2 * torch.arange(self.n_quantiles) + 1) / (2*self.n_quantiles)).view(1, 1, -1).to(self.device)

        self.explorer = EpsilonGreedy(config)
        self.replay_buffer = ReplayBuffer(config)

        # Action mesh
        self._generate_action_mesh()

        # Critic network
        self.critic = CriticMLP(self.s_dim, self.a_mesh_dim * self.n_quantiles, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic = CriticMLP(self.s_dim, self.a_mesh_dim * self.n_quantiles, hidden_dim_lst, F.silu).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.loss_lst = ['Critic loss']

    def generate_non_uniform_mesh(self, n):
        # Create a uniform grid in the interval [-1, 1]
        uniform_mesh = np.linspace(-0.999, 0.999, n)

        # Apply a non-linear transformation (e.g., tanh) to make the mesh denser near the origin
        non_uniform_mesh = np.arctanh(uniform_mesh)  # Adjust scaling factor

        non_uniform_mesh = non_uniform_mesh / np.max(np.abs(non_uniform_mesh))

        # Ensure the origin is included in the grid
        if 0 not in non_uniform_mesh:
            non_uniform_mesh = np.append(non_uniform_mesh, 0)
            non_uniform_mesh = np.sort(non_uniform_mesh)

        return non_uniform_mesh

    def _generate_action_mesh(self):
        n_per_dim = min(21, max(11, int(self.max_n_action_grid ** (1 / self.a_dim))))
        self.single_dim_mesh = self.generate_non_uniform_mesh(n_per_dim)

        # Generate action mesh and mesh index for discrete action space
        single_dim_mesh = np.array(self.single_dim_mesh)

        self.a_mesh_dim = len(single_dim_mesh) ** self.a_dim
        self.a_mesh_idx = np.arange(self.a_mesh_dim).reshape(*[len(single_dim_mesh) for _ in range(self.a_dim)])  # (M, M, .., M)
        self.a_mesh = np.stack(np.meshgrid(*[single_dim_mesh for _ in range(self.a_dim)]))  # (A, M, M, ..., M)

        self.explorer.mesh_size = self.a_mesh_dim

    def ctrl(self, state):
        with torch.no_grad():
            state = torch.tensor(state.T, dtype=torch.float32, device=self.device)
            q_values = self.critic(state).reshape(-1, self.a_mesh_dim, self.n_quantiles).mean(dim=2)

        _, action_idx = torch.min(q_values, dim=1, keepdim=True)
        action_idx = action_idx.cpu().numpy()
        action_idx = self.explorer.sample(action_idx)
        action = self._idx2action(action_idx)

        return action

    def _idx2action(self, idx):
        if self.a_dim == 1:
            idx = idx[0]
        # Get action values from indexes
        mesh_idx = np.where(self.a_mesh_idx == idx)
        action = np.array([self.a_mesh[i][mesh_idx] for i in range(self.a_dim)])

        return action

    def add_experience(self, experience):
        state, action, reward, next_state, done, deriv = experience
        action_idx = self._action2idx(action)

        self.replay_buffer.add((state, action_idx, reward, next_state, done, deriv))

    def _action2idx(self, action):
        # Find nearest action grid values
        import math
        def find_nearest(array, value):
            idx = np.searchsorted(array, value, side="left")
            if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
                return array[idx - 1]
            else:
                return array[idx]

        nrst_action_lst = [find_nearest(self.single_dim_mesh, action[i].item()) for i in range(self.a_dim)]

        # Get indexes from action values
        action2idx_lst = [self.a_mesh[i] == nrst_action_lst[i] for i in range(self.a_dim)]

        idx_lst = action2idx_lst[0]
        for i in range(1, len(action2idx_lst)):
            idx_lst = idx_lst & action2idx_lst[i]
        idx = np.where(idx_lst)
        mesh_idx = self.a_mesh_idx[idx].reshape(-1, 1)

        return mesh_idx

    def warm_up_train(self):
        pass

    def train(self):
        # Replay buffer sample
        sample = self.replay_buffer.sample()
        states = sample['states']
        action_indices = sample['actions']
        rewards = sample['rewards']
        next_states = sample['next_states']
        dones = sample['dones']

        with torch.no_grad():
            next_q = self.target_critic(next_states).reshape(-1, self.a_mesh_dim, self.n_quantiles)
            _, next_action_indices = torch.min(next_q.mean(dim=2), dim=1, keepdim=True)
            next_action_indices = next_action_indices.unsqueeze(-1).repeat(1, 1, self.n_quantiles)
            next_q = torch.gather(next_q, dim=1, index=next_action_indices)
            target_q = rewards.unsqueeze(2) + self.gamma * next_q * (1 - dones.unsqueeze(2))

        current_q = self.critic(states).reshape(-1, self.a_mesh_dim, self.n_quantiles)
        action_indices = action_indices.unsqueeze(-1).repeat(1, 1, self.n_quantiles).long()
        current_q = torch.gather(current_q, dim=1, index=action_indices)

        # Compute quantile Huber loss
        huber_loss = F.huber_loss(current_q, target_q, reduction='none')
        error = target_q - current_q
        critic_loss = torch.mean((self.quantile_taus - (error < 0).float()).abs() * huber_loss)

        # Optimize the critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
        self.critic_optimizer.step()

        # Soft update
        for to_model, from_model in zip(self.target_critic.parameters(), self.critic.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        loss = np.array([critic_loss.cpu().detach().item()])

        return loss
    
    def quantile_huber_loss(self, current_q, target_q):
        huber_loss = F.huber_loss(current_q, target_q, reduction='none')
        error = target_q - current_q
        loss = torch.mean((self.quantile_taus - (error < 0).float()).abs() * huber_loss)
        return loss

    def save(self, path, file_name):
        torch.save(self.critic.state_dict(), os.path.join(path, file_name + '_critic.pt'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, file_name + '_critic_optimizer.pt'))

    def load(self, path, file_name):
        self.critic.load_state_dict(torch.load(os.path.join(path, file_name + '_critic.pt')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic_optimizer.pt')))
        self.target_critic = copy.deepcopy(self.critic)
