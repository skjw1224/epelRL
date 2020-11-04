import utils
import torch
import numpy as np

class PID(object):
    def __init__(self, config):
        self.env = config.environment

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.o_dim = self.env.o_dim

    def ctrl(self, epi, step, x, u):
        if step == 0:
            self.ei = np.zeros([self.o_dim, 1])
        ref = utils.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax)
        Kp = 2 * np.ones([self.a_dim, self.o_dim])
        Ki = 0.1 * np.ones([self.a_dim, self.o_dim])

        y = self.env.y_fnc(x, u, self.env.param_real, self.env.param_sigma_prior, np.zeros([self.env.p_dim, 1])).full()
        u = Kp @ (y - ref) + Ki @ self.ei
        print(u)

        self.ei = self.ei + (y - ref)
        return u

    def add_experience(self, *single_expr):
        pass