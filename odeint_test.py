import unittest
from CSTR.ENVIRONMENT import CstrEnv
import torch
import sys
import numpy as np

device = 'cuda'


class OdeintTest(unittest.TestCase):

    def test_ode(self):
        env = CstrEnv(device)
        s_dim = env.s_dim
        a_dim = env.a_dim
        o_dim = env.o_dim
        x, y, prev_u = env.reset()
        t = env.time
        # t, x, y = env.reset()
        trajectory = torch.zeros([1, s_dim + o_dim + a_dim + 1], device=device)  # s + o + a + r
        MAX_EP_STEPS = 2
        time_track = torch.tensor([0.], device=device)

        for i in range(MAX_EP_STEPS):
            terminal = False
            delu = torch.tensor([[.1, .1]], device=device)
            print(t)
            u = prev_u + delu
            x2, y2, u, r, terminal, dx = env.step(delu)
            t2 = env.time
            t, x = t2, x2

        return True


if __name__ == '__main__':
    unittest.main()