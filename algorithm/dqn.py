import os
import copy
import torch
import torch.nn.functional as F
import numpy as np

from .algorithm import Algorithm
from replay_buffer.replay_buffer import ReplayBuffer


class DQN(Algorithm):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim

        # Hyperparameters
        self.h_nodes = self.config.hyperparameters['hidden_nodes']
        self.explore_epi_idx = self.config.hyperparameters['explore_epi_idx']
        self.buffer_size = self.config.hyperparameters['buffer_size']
        self.minibatch_size = self.config.hyperparameters['minibatch_size']
        self.learning_rate = self.config.hyperparameters['learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.tau = self.config.hyperparameters['tau']
        self.single_dim_mesh = self.config.hyperparameters['single_dim_mesh']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.minibatch_size)

        # Action mesh
        self.a_mesh_dim, self.a_mesh, self.a_mesh_idx = self.generate_action_mesh()

        # Critic network (Q network)
        self.critic = self.approximator(self.s_dim, self.a_mesh_dim, self.h_nodes).to(self.device)  # s --> a
        self.target_critic = self.approximator(self.s_dim, self.a_mesh_dim, self.h_nodes).to(self.device)  # s --> a
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        for to_model, from_model in zip(self.target_critic.parameters(), self.critic.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.loss_lst = ['Critic loss']

    def ctrl(self, epi, step, s, a):
        if epi < self.explore_epi_idx:
            a_idx = self.choose_action(s)
            a_idx = self.explorer.sample(epi, step, a_idx)
        else:
            a_idx = self.choose_action(s)

        return a_idx

    def choose_action(self, s):
        # Numpy to torch
        s = torch.from_numpy(s.T).float().to(self.device)  # (B, 1)

        self.critic.eval()
        with torch.no_grad():
            a_idx = self.critic(s).min(-1)[1].unsqueeze(1)
        self.critic.train()

        # Torch to Numpy
        a_idx = a_idx.cpu().detach().numpy()

        return a_idx

    def generate_action_mesh(self):
        num_grid = len(self.single_dim_mesh)
        a_mesh_dim = num_grid ** self.a_dim
        single_dim_mesh = np.array(self.single_dim_mesh)
        a_mesh = np.stack(np.meshgrid(*[single_dim_mesh for _ in range(self.a_dim)]))  # (A, M, M, ..., M)
        a_mesh_idx = np.arange(a_mesh_dim).reshape(*[num_grid for _ in range(self.a_dim)])  # (M, M, .., M)

        return a_mesh_dim, a_mesh, a_mesh_idx

    def action_idx2mesh(self, a_idx):
        env_a_dim = len(self.a_mesh)

        mesh_idx = (self.a_mesh_idx == a_idx).nonzero()
        a = np.array([self.a_mesh[i, :][tuple(mesh_idx)] for i in range(env_a_dim)])

        return a

    def add_experience(self, *single_expr):
        s, a_idx, r, s2, is_term = single_expr
        self.replay_buffer.add(*[s, a_idx, r, s2, is_term])

    def train(self):
        if len(self.replay_buffer) > 0:
            s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample()

            q_batch = self.critic(s_batch).gather(1, a_batch.long())
            q2_batch = self.target_critic(s2_batch).detach().min(-1)[0].unsqueeze(1) * (1 - term_batch)
            q_target_batch = r_batch + q2_batch

            critic_loss = F.mse_loss(q_batch, q_target_batch)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
            self.critic_optimizer.step()

            """Updates the target network in the direction of the local network but by taking a step size
            less than one so the target network's parameter values trail the local networks. This helps stabilise training"""

            for to_model, from_model in zip(self.target_critic.parameters(), self.critic.parameters()):
                to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

            critic_loss = critic_loss.cpu().detach().numpy().item()
            loss = np.array([critic_loss])
        else:
            loss = np.array([0.])

        return loss

    def save(self, path, file_name):
        torch.save(self.critic.state_dict(), os.path.join(path, file_name + '_critic.pt'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, file_name + '_critic_optimizer.pt'))

    def load(self, path, file_name):
        self.critic.load_state_dict(torch.load(os.path.join(path, file_name + '_critic.pt')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic_optimizer.pt')))
        self.target_critic = copy.deepcopy(self.critic)