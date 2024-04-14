import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .trpo import TRPO


class PPO(TRPO):
    def __init__(self, config):
        TRPO.__init__(self, config)
        self.actor_lr = config.actor_lr
        self.clip_epsilon = config.clip_epsilon

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, eps=self.adam_eps, weight_decay=self.l2_reg)

    def train(self):
        # Replay buffer sample
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_sequence()

        # Compute returns and advantages
        returns, advantages = self._gae_estimation(states, rewards, next_states, dones)

        # Train actor and critic network with surrogate loss
        surrogate_loss = self._compute_surrogate_loss(states, actions, advantages)
        actor_loss = self._actor_update(surrogate_loss)
        critic_loss = self._critic_update(states, returns)
        loss = np.array([critic_loss, actor_loss])

        # Clear replay buffer
        self.replay_buffer.clear()

        return loss

    def _compute_surrogate_loss(self, states, actions, advantages):
        # Compute surrogate loss and KL divergence
        with torch.no_grad():
            distribution_old, log_probs_old = self.actor.get_log_prob(states, actions)
        distribution_new, log_probs_new = self.actor.get_log_prob(states, actions)

        ratio = torch.exp(log_probs_new - log_probs_old.detach())
        surrogate_loss1 = advantages.detach() * ratio
        surrogate_loss2 = advantages.detach() * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surrogate_loss = torch.min(surrogate_loss1, surrogate_loss2).mean()

        return surrogate_loss

    def _actor_update(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_mag)
        self.actor_optimizer.step()

        return loss.detach().cpu().item()

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


