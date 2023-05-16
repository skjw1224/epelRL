import torch
from torch.distributions import kl_divergence
from trpo import TRPO


class PPO(TRPO):
    def _compute_surrogate_loss_and_kl_divergence(self, states, actions, advantages):
        surrogate_loss = None
        kl_div = None

        return surrogate_loss, kl_div

