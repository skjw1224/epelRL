import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_rbf as rbf
import random
import numpy as np
import os
my_CSTR = os.getcwd()

from replay_buffer import ReplayBuffer

SIGMA = 0.1

INITIAL_POLICY_INDEX = 1
BASIS_NUMBERS = 40
BASIS_FCNS = rbf.gaussian


class PoWER(object):
    def __init__(self, env, device):
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.env_epi_length = env.nT

        self.device = device

        self.replay_buffer = ReplayBuffer(env, device, buffer_size=self.env_epi_length, batch_size=self.env_epi_length)
        self.initial_ctrl = InitialControl(env, device)

        self.rbfnet = rbf.RBF(self.s_dim, BASIS_NUMBERS, BASIS_FCNS)
        self.f_dim = BASIS_NUMBERS

        self.theta = torch.randn([self.f_dim, self.a_dim], device=device)
        self.S = SIGMA * torch.ones([self.f_dim, ], device=device)

    def ctrl(self, epi, step, x, u):
        if epi < INITIAL_POLICY_INDEX:
            a_val = self.initial_ctrl.controller(epi, step, x, u)
            a_val.detach()
            self.exp_schedule(epi, step)
        else:
            a_val = self.choose_action(epi, step, x)

        a_val = torch.clamp(a_val, -1., 1.)

        if len(self.replay_buffer) == self.env_epi_length:
            self.train()
        return a_val

    def choose_action(self, epi, step, x):
        phi_val = self.rbfnet(x) # (1, F)

        epsilon = self.exp_schedule(epi, step)
        action = phi_val @ (self.theta + epsilon) # (1, F) @ (F, A)
        return action

    def add_experience(self, *single_expr):
        x, u, r, x2, term = single_expr
        self.replay_buffer.add(*[x, u, r, x2, term])

        if term is True: # In on-policy method, clear buffer when episode ends
            self.replay_buffer.clear()

    def exp_schedule(self, epi, step):

        if step == 0:
            self.S = torch.mean((self.theta)**2, 1) * 0.1
            eps_dist = torch.distributions.Normal(torch.zeros([1, self.f_dim]), torch.sqrt(self.S))
            self.epsilon_traj = eps_dist.sample([self.env_epi_length, self.a_dim]).view(self.env_epi_length, -1, self.a_dim) # (T, F, A)

        epsilon = self.epsilon_traj[step, :]
        return epsilon

    def reweight(self):
        s_traj, a_traj, r_traj, s2_traj, term_traj = self.replay_buffer.sample_sequence() # T-number sequence
        phi_traj = self.rbfnet(s_traj) # (T, F)

        # Compute cumulative return through trajectory
        q_traj = [r_traj[-1]]
        for i in range(len(self.replay_buffer)):
            q_traj.append(r_traj[-i-1] + q_traj[-1])

        q_traj.reverse()
        q_traj = torch.stack(q_traj[:-1]) # (T, 1)

        # Compute W function through trajectory
        W_traj = []
        for i in range(len(self.replay_buffer)):
            phi_i = phi_traj[i, :].unsqueeze(1) # (F, 1)
            W = phi_i @ phi_i.T / (phi_i.T @ torch.diag(self.S) @ phi_i) # [F, F]
            W_traj.append(W)
        W_traj = torch.stack(W_traj) # [T, F, F]

        # Sum up w.r.t time index
        # Numerator
        theta_num = (W_traj @ self.epsilon_traj).permute(1, 2, 0) @ q_traj # [F, A, T] @ [T, 1]
        theta_num.squeeze_(-1) # [F, A]

        # Denominator
        theta_den = W_traj.T @ q_traj # [F, F, T] @ [T, 1]
        theta_den.squeeze_(-1) # [F, F]
        try:
            theta_den_chol = torch.cholesky(theta_den + 1E-4 * torch.eye(self.f_dim))
        except RuntimeError:
            theta_den_chol = torch.cholesky(theta_den + 1E-2 * torch.eye(self.f_dim))
        del_theta = torch.cholesky_solve(theta_num, theta_den_chol) # [F, A]

        return del_theta

    def train(self):
        del_theta = self.reweight()
        self.theta = self.theta + del_theta

from pid import PID
class InitialControl(object):
    def __init__(self, env, device):
        self.pid = PID(env, device)

    def controller(self, epi, step, x, u):
        return self.pid.ctrl(epi, step, x, u)