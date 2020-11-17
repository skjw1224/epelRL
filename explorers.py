import numpy as np
import random
import copy

class OU_Noise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.a_dim = self.env.a_dim

        self.mu0 = self.config.hyperparameters['ou_mu0']
        self.theta = self.config.hyperparameters['ou_theta']
        self.sigma = self.config.hyperparameters['ou_sigma']

        self.eps0 = self.config.hyperparameters['eps_greedy']
        self.epi_denom = self.config.hyperparameters['eps_greedy_denom']

        self.mu = self.mu0 * np.ones([self.a_dim, 1])

        random.seed(123)

    def exp_schedule(self, epi):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu
        self.epsilon = self.eps0 / (1. + (epi / self.epi_denom))

    def sample(self, epi, step, u_nom):
        if step == 0:
            self.exp_schedule(epi)

        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.a_dim, 1)
        self.state += dx
        noise = self.state * self.epsilon
        u_exp = noise + u_nom

        return u_exp

class E_greedy(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.a_dim = self.env.a_dim

        self.eps0 = self.config.hyperparameters['eps_greedy']
        self.epi_denom = self.config.hyperparameters['eps_greedy_denom']

    def exp_schedule(self, epi):
        self.eps = self.eps0 / (1. + (epi / self.epi_denom))

    def sample(self, epi, step, u_nom):
        if step == 0:
            self.exp_schedule(epi)

        if np.random.random() <= self.eps:
            u_exp = np.random.randint(low=0, high=self.a_dim, size=[1, 1])
        else:
            u_exp = u_nom

        return u_exp

class Gaussian_noise(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.a_dim = self.env.a_dim

        self.eps_decay_rate = self.config.hyperparameters['eps_decay_rate']

        random.seed(123)

    def exp_schedule(self, epi):
        self.eps = self.eps_decay_rate ** epi * np.random.randn(self.a_dim, 1)

    def sample(self, epi, step, u_nom):
        if step == 0:
            self.exp_schedule(epi)

        u_exp = self.eps + u_nom

        return u_exp