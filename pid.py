import utils
import torch

class PID(object):
    def __init__(self, config):
        self.env = config.environment

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim

    def ctrl(self, epi, step, x, u):
        if step == 0:
            self.ei = torch.zeros([1, self.s_dim])
        ref = utils.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax)
        Kp = 2 * torch.ones([self.a_dim, self.s_dim], requires_grad=False)
        Ki = 0.1 * torch.ones([self.a_dim, self.s_dim], requires_grad=False)
        u = Kp @ (x - ref).T + Ki @ self.ei.T
        u = u.T

        self.ei = self.ei + (x - ref)
        return u

    def add_experience(self, *single_expr):
        pass