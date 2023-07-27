import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from trpo import TRPO


class PPO(TRPO):
    def __init__(self, config):
        TRPO.__init__(self, config)
        self.actor_learning_rate = config.hyperparameters['actor_learning_rate']
        self.clip_epsilon = config.hyperparameters['clip_epsilon']

        self.actor_net_opt = optim.Adam(self.actor_net.parameters(), lr=self.actor_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

    def train(self):
        # Replay buffer sample
        s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample_sequence()

        # Compute returns and advantages
        returns, advantages = self._gae_estimation(s_batch, r_batch, term_batch)

        # Train actor and critic network with surrogate loss
        surrogate_loss = self._compute_surrogate_loss(s_batch, a_batch, advantages)
        actor_loss = self._actor_update(surrogate_loss)
        critic_loss = self._critic_update(s_batch, returns)
        loss = np.array([critic_loss, actor_loss])

        # Clear replay buffer
        self.replay_buffer.clear()

        return loss

    def _compute_surrogate_loss(self, states, actions, advantages):
        # Compute surrogate loss and KL divergence
        log_probs_new, _ = self._get_log_probs(states, actions, True)
        log_probs_old, _ = self._get_log_probs(states, actions, False)

        ratio = torch.exp(log_probs_new - log_probs_old.detach())
        surrogate_loss1 = advantages * ratio
        surrogate_loss2 = advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        surrogate_loss = torch.min(surrogate_loss1, surrogate_loss2).mean()

        return surrogate_loss

    def _actor_update(self, loss):
        self.actor_net_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.grad_clip_mag)
        self.actor_net_opt.step()

        return loss.detach().cpu().item()




