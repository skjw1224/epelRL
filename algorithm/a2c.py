import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from .base_algorithm import Algorithm
from replay_buffer.replay_buffer import ReplayBuffer
from network.network import ActorMlp, CriticMLP


class A2C(Algorithm):
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

        config.buffer_size = self.nT
        config.batch_size = self.nT
        self.replay_buffer = ReplayBuffer(config)

        # Critic network (Value network)
        self.critic = CriticMLP(self.s_dim, 1, hidden_dim_lst, F.silu).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor network
        self.actor = ActorMlp(self.s_dim, self.a_dim, hidden_dim_lst, F.silu).to(self.device)
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=self.actor_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.loss_lst = ['Critic loss', 'Actor loss']

    def ctrl(self, state):
        state = torch.from_numpy(state.T).float().to(self.device)
        action, _ = self.actor(state, deterministic=False, reparam_trick=False, return_log_prob=False)
        action = np.clip(action.T.cpu().detach().numpy(), -1., 1.)

        return action

    def add_experience(self, *single_expr):
        state, action, reward, next_state, done = single_expr
        self.replay_buffer.add(*[state, action, reward, next_state, done])

    def _get_log_prob(self, s_batch, a_batch):
        a_pred = self.actor(s_batch)
        mean, log_std = a_pred[:, :self.a_dim], a_pred[:, self.a_dim:]
        std = torch.exp(log_std)
        distribution = Normal(mean, std)
        log_prob = distribution.log_prob(a_batch)

        return log_prob

    def train(self):
        # Replay buffer sample
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_sequence()

        log_prob_traj = self.actor.get_log_prob(states, actions)

        v_target_traj = []

        if dones[-1]:  # When Final value of sequence is terminal sample
            v_target_traj.append(rewards[-1])  # Append terminal cost
        else:  # When Final value of sequence is path sample
            v_target_traj.append(self.critic(next_states[-1]))  # Append n-step bootstrapped q-value

        for i in range(len(states)):
            v_target_traj.append(rewards[-i-1] + v_target_traj[-1])

        v_target_traj.reverse()
        v_target_traj = torch.stack(v_target_traj[:-1])
        v_target_traj.detach()

        v_traj = self.critic(states)
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
