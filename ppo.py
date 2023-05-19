import torch
from torch.distributions import kl_divergence
from trpo import TRPO


class PPO(TRPO):
    def __init__(self, config):
        TRPO.__init__(self, config)
        self.clip_epsilon = config.hyperparameters['clip_epsilon']

    def train(self, step):
        if step == self.nT - 1:
            # Replay buffer sample
            s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample_sequence()

            # Compute returns and advantages
            returns, advantages = self._gae_estimation(s_batch, r_batch, term_batch)

            # TODO: actor and critic update method should be changed

            # Clear replay buffer
            self.replay_buffer.clear()

            loss = actor_loss + critic_loss

        else:
            loss = 0.

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


