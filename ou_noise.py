import numpy as np
import random
import copy

class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, config, mu=0., theta=0.15, sigma=0.2):
        self.config = config
        self.env = self.config.environment
        self.a_dim = self.env.a_dim

        self.mu = mu * np.ones([self.a_dim, 1])

        mu0 = self.config.hyperparameters

        self.theta = theta
        self.sigma = sigma
        random.seed(123)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state