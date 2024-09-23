import os
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .base_algorithm import Algorithm
from network.nn import ActorMlp, CriticMLP
from utility.buffer import ReplayBuffer
from utility.explorers import OUNoise


class GDHP(Algorithm):
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
        self.costate_lr = self.config['costate_lr']
        self.adam_eps = self.config['adam_eps']
        self.l2_reg = self.config['l2_reg']
        self.grad_clip_mag = self.config['grad_clip_mag']
        self.tau = self.config['tau']

        self.explorer = OUNoise(config)
        self.replay_buffer = ReplayBuffer(config)

        # Critic network
        self.critic = CriticMLP(self.s_dim, 1, hidden_dim_lst, F.silu).to(self.device)  # s --> 1
        self.target_critic = CriticMLP(self.s_dim, 1, hidden_dim_lst, F.silu).to(self.device)  # s --> 1
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor network
        self.actor = ActorMlp(self.s_dim, self.a_dim, hidden_dim_lst, F.silu).to(self.device)  # s --> a
        self.target_actor = ActorMlp(self.s_dim, self.a_dim, hidden_dim_lst, F.silu).to(self.device)  # s --> a
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Costate network
        self.costate = CriticMLP(self.s_dim, self.s_dim, hidden_dim_lst, F.silu).to(self.device)  # s --> s
        self.target_costate = CriticMLP(self.s_dim, self.s_dim, hidden_dim_lst, F.silu).to(self.device)  # s --> s
        self.target_costate = copy.deepcopy(self.costate)
        self.costate_optimizer = optim.Adam(self.costate.parameters(), lr=self.costate_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.loss_lst = ['Critic loss', 'Costate loss', 'Actor loss']

    def ctrl(self, state):
        with torch.no_grad():
            state = torch.tensor(state.T, dtype=torch.float32, device=self.device)
            action = self.actor(state, deterministic=True, reparam_trick=False, return_log_prob=False).cpu().numpy()

        action = self.explorer.sample(action.T)
        action = np.clip(action, -1., 1.)

        return action

    def add_experience(self, experience):
        self.replay_buffer.add(experience)

    def train(self):
        # Replay buffer sample
        sample = self.replay_buffer.sample()
        states = sample['states']
        actions = sample['actions']
        rewards = sample['rewards']
        next_states = sample['next_states']
        dones = sample['dones']
        derivs = sample['derivs']
        dfdx_batch, dfdu_batch, dcdx_batch, d2cdu2inv_batch = derivs[0], derivs[1], derivs[2], derivs[7]

        # Critic Train
        with torch.no_grad():
            next_q = self.target_critic(next_states)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.critic(states)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_mag)
        self.critic_optimizer.step()

        # Costate Train
        with torch.no_grad():
            l2_batch = self.target_costate(next_states) * (1 - dones)
            l_target_batch = (dcdx_batch.permute(0, 2, 1) + l2_batch.unsqueeze(1) @ dfdx_batch).squeeze(1)  # (B, S)
        
        l_batch = self.costate(states)
        costate_loss = F.mse_loss(l_batch, l_target_batch)

        self.costate_optimizer.zero_grad()
        costate_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.costate.parameters(), self.grad_clip_mag)
        self.costate_optimizer.step()

        # Actor Train
        with torch.no_grad():
            a_target_batch = torch.clamp((-0.5 * l2_batch.unsqueeze(1) @ dfdu_batch @ d2cdu2inv_batch), -1., 1.).squeeze(1)
        
        actions = self.actor(states, deterministic=True, reparam_trick=False, return_log_prob=False)
        actor_loss = F.mse_loss(actions, a_target_batch)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_mag)
        self.actor_optimizer.step()

        # Soft update
        for to_model, from_model in zip(self.target_critic.parameters(), self.critic.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        for to_model, from_model in zip(self.target_costate.parameters(), self.costate.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        for to_model, from_model in zip(self.target_actor.parameters(), self.actor.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        critic_loss = critic_loss.detach().cpu().item()
        costate_loss = costate_loss.detach().cpu().item()
        actor_loss = actor_loss.detach().cpu().item()
        loss = np.array([critic_loss, costate_loss, actor_loss])

        return loss

    def save(self, path, file_name):
        torch.save(self.critic.state_dict(), os.path.join(path, file_name + '_critic.pt'))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(path, file_name + '_critic_optimizer.pt'))

        torch.save(self.actor.state_dict(), os.path.join(path, file_name + '_actor.pt'))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(path, file_name + '_actor_optimizer.pt'))

        torch.save(self.costate.state_dict(), os.path.join(path, file_name + '_costate.pt'))
        torch.save(self.costate_optimizer.state_dict(), os.path.join(path, file_name + '_costate_optimizer.pt'))

    def load(self, path, file_name):
        self.critic.load_state_dict(torch.load(os.path.join(path, file_name + '_critic.pt')))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_critic_optimizer.pt')))
        self.target_critic = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(os.path.join(path, file_name + '_actor.pt')))
        self.actor_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_actor_optimizer.pt')))
        self.target_actor = copy.deepcopy(self.actor)

        self.costate.load_state_dict(torch.load(os.path.join(path, file_name + '_costate.pt')))
        self.costate_optimizer.load_state_dict(torch.load(os.path.join(path, file_name + '_costate_optimizer.pt')))
        self.target_costate = copy.deepcopy(self.costate)
        