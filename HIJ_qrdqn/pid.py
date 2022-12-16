import utils
import torch

class PID(object):
    def __init__(self, env, device, reverse_mode=True, K_val=2., Ki_val=.1):
        """Accept K_val or Ki_val as an integer, or (A, O) dimension tensor."""
        self.env = env
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.o_dim = env.o_dim
        self.ei = torch.zeros([1, self.o_dim])
        self.dt = env.dt
        self.K = []
        for value in (K_val, Ki_val):
            if reverse_mode:
                value = -value
            if torch.is_tensor(value):
                K = value
            else:
                K = value * torch.ones([self.a_dim, self.o_dim], requires_grad=False)
            self.K.append(K)

    def pid_ctrl(self, step, t, x):
        if step == 0:
            self.reset()
        ref = utils.scale(self.env.ref_traj(t), self.env.ymin, self.env.ymax)
        error = x[:, 1] - ref
        # error = error.reshape([1, -1])
        Kp, Ki = self.K
        K_prev = torch.zeros([self.a_dim, self.s_dim], requires_grad=False)
        K_prev[-2, -2] = 1.
        K_prev[-1, -1] = 1.
        del_u = Kp @ error.T + self.dt * Ki @ self.ei.T - K_prev @ x.T    # to get the del_u for action execution
        del_u = del_u.T

        self.ei = self.ei + error
        return del_u

    def pid_step_response(self, step, t, x):
        if step == 0:
            self.reset()


    def reset(self):
        self.ei = torch.zeros([1, self.o_dim])