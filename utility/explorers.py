import numpy as np


class OUNoise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, config):
        self.a_dim = config.a_dim
        self.nT = config.nT

        self.mu0 = 0
        self.theta = 0.15
        self.sigma = 0.2
        self.eps0 = 0.3
        self.epi_denom = 1

        self.mu = self.mu0 * np.ones([self.a_dim, 1])

        self.step = 0
        self.episode = 0
        self.exp_schedule()

    def exp_schedule(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu
        self.epsilon = self.eps0 / (1. + (self.episode / self.epi_denom))

    def sample(self, u_nom):
        """Update internal state and return it as a noise sample."""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.a_dim, 1)
        self.state += dx
        noise = self.state * self.epsilon
        u_exp = noise + u_nom

        self.step += 1
        if self.step >= self.nT:
            self.step = 0
            self.episode += 1
            self.exp_schedule()

        return u_exp


class EpsilonGreedy(object):
    def __init__(self, config):
        self.a_dim = config.a_dim

        n_per_dim = max(11, int(config.max_n_action_grid ** (1 / self.a_dim)))
        self.mesh_size = n_per_dim ** self.a_dim
        self.nT = config.nT

        self.eps0 = 0.3
        self.epi_denom = 1

        self.step = 0
        self.episode = 0
        self.exp_schedule()

    def exp_schedule(self):
        self.epsilon = self.eps0 / (1. + (self.episode / self.epi_denom))

    def sample(self, u_nom):
        if np.random.random() <= self.epsilon:
            u_exp = np.random.randint(low=0, high=self.mesh_size, size=[1, 1])
        else:
            u_exp = u_nom

        self.step += 1
        if self.step >= self.nT:
            self.step = 0
            self.episode += 1
            self.exp_schedule()

        return u_exp


class GaussianNoise(object):
    def __init__(self, config):
        self.a_dim = config.a_dim
        self.nT = config.nT

        self.eps_decay_rate = 0.99

        self.step = 0
        self.episode = 0
        self.exp_schedule()

    def exp_schedule(self):
        self.epsilon = self.eps_decay_rate ** self.episode * np.random.randn(self.a_dim, 1)

    def sample(self, u_nom):
        u_exp = self.epsilon + u_nom

        self.step += 1
        if self.step >= self.nT:
            self.step = 0
            self.episode += 1
            self.exp_schedule()

        return u_exp
