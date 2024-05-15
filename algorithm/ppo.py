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
        sample = self.rollout_buffer.sample()
        states = sample['states']
        actions = sample['actions']
        rewards = sample['rewards']
        next_states = sample['next_states']
        dones = sample['dones']

        # Compute generalized advantage estimations (GAE) and returns
        values = self.critic(states)
        next_values = self.critic(next_states)
        delta = rewards + self.gamma * next_values * (1 - dones) - values

        advantages = torch.zeros_like(rewards)
        advantage = 0
        for t in reversed(range(len(self.rollout_buffer))):
            advantage = delta[t] + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage
        returns = advantages + values
        
        if advantages.shape[0] > 1:  # advantage normalization only if the batch size is bigger than 1
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Train actor and critic network with surrogate loss
        surrogate_loss = self._compute_surrogate_loss(states, actions, advantages)
        actor_loss = self._actor_update(surrogate_loss)
        critic_loss = self._critic_update(states, returns)
        loss = np.array([critic_loss, actor_loss])

        # Clear replay buffer
        self.rollout_buffer.reset()

        return loss

    def _compute_surrogate_loss(self, states, actions, advantages):
        # Compute surrogate loss and KL divergence
        with torch.no_grad():
            _, log_probs_old = self.old_actor.get_log_prob(states, actions)
        _, log_probs_new = self.actor.get_log_prob(states, actions)

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


