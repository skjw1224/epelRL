import torch
from torch.distributions import Normal
import numpy as np

from replay_buffer import ReplayBuffer


class PoWER(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = 'cpu'

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        # Hyperparameters
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.rbf_dim = self.config.hyperparameters['rbf_dim']
        self.rbf_type = self.config.hyperparameters['rbf_type']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT, batch_size=self.nT)

        # Actor network
        self.actor_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type)
        self.theta = torch.randn([self.rbf_dim, self.a_dim])
        self.actor_std = 0.1 * torch.ones([1, self.rbf_dim])

    def ctrl(self, epi, step, s, a):
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.ctrl(epi, step, s, a)
            a_val = self.explorer.sample(epi, step, a_nom)
        else:
            a_val = self._choose_action(s)

        a_val = np.clip(a_val, -1., 1.)

        return a_val

    def _choose_action(self, s):
        # numpy to torch
        s = torch.from_numpy(s.T).float().to(self.device)

        phi = self.actor_net(s)  # (1, F)
        epsilon = self.exp_schedule(epi, step)
        a = phi @ (self.theta + epsilon)  # (1, F) @ (F, A)

        # torch to numpy
        a = a.T.cpu().detach().numpy()

        return a

    def add_experience(self, *single_expr):
        s, a, r, s2, is_term = single_expr
        self.replay_buffer.add(*[s, a, r, s2, is_term])

        if is_term is True:  # In on-policy method, clear buffer when episode ends
            self.replay_buffer.clear()


    def exp_schedule(self, epi, step):
        if step == 0:
            self.S = torch.mean((self.theta)**2, 1) * 0.1
            eps_dist = torch.distributions.Normal(torch.zeros([1, self.rbf_dim]), torch.sqrt(self.S))
            self.epsilon_traj = eps_dist.sample([self.nT, self.a_dim]).view(self.nT, -1, self.a_dim)  # (T, F, A)

        epsilon = self.epsilon_traj[step, :]
        return epsilon

    def train(self, step):
        if step == self.nT:
            s_traj, a_traj, r_traj, s2_traj, term_traj = self.replay_buffer.sample_sequence()
            eps_traj = self._sample()
            q_traj = self._estimate(r_traj)
            w_traj = self._reweight(s_traj)
            loss = self._update(q_traj, w_traj)
        else:
            loss = 0.

        return loss

    def _sample(self):
        eps_distribution = Normal(torch.zeros([1, self.rbf_dim]), self.actor_std)
        eps_traj = eps_distribution.sample()

        return eps_traj

    def _estimate(self, r_traj):
        q_traj = [r_traj[-1]]
        for t in range(self.nT):
            q_traj.append(r_traj[-t-1] + q_traj[-1])

        q_traj.reverse()
        q_traj = np.array(q_traj[:-1])

        return q_traj

    def _reweight(self, s_traj):
        w_traj = []
        phi_traj = self.actor_net(s_traj)
        for t in range(self.nT):
            phi = phi_traj[t, :].unsqueeze(1)
            sigma = np.diag(self.actor_std)
            w = phi @ phi.T / (phi.T @ sigma @ phi)
            w_traj.append(w)
        w_traj = np.array(w_traj)

        return w_traj

    def _update(self, q_traj, w_traj):
        # Update policy

        # Update standard deviation

        pass

    def reweight(self):
        s_traj, a_traj, r_traj, s2_traj, term_traj = self.replay_buffer.sample_sequence()  # T-number sequence
        phi_traj = self.actor_net(s_traj)  # (T, F)

        # Compute cumulative return through trajectory
        q_traj = [r_traj[-1]]
        for i in range(len(self.replay_buffer)):
            q_traj.append(r_traj[-i-1] + q_traj[-1])

        q_traj.reverse()
        q_traj = torch.stack(q_traj[:-1]) # (T, 1)

        # Compute W function through trajectory
        W_traj = []
        for i in range(len(self.replay_buffer)):
            phi_i = phi_traj[i, :].unsqueeze(1)  # (F, 1)
            W = phi_i @ phi_i.T / (phi_i.T @ torch.diag(self.S) @ phi_i)  # [F, F]
            W_traj.append(W)
        W_traj = torch.stack(W_traj)  # [T, F, F]

        # Sum up w.r.t time index
        # Numerator
        theta_num = (W_traj @ self.epsilon_traj).permute(1, 2, 0) @ q_traj  # [F, A, T] @ [T, 1]
        theta_num.squeeze_(-1)  # [F, A]

        # Denominator
        theta_den = W_traj.T @ q_traj  # [F, F, T] @ [T, 1]
        theta_den.squeeze_(-1)  # [F, F]
        try:
            theta_den_chol = torch.cholesky(theta_den + 1E-4 * torch.eye(self.rbf_dim))
        except RuntimeError:
            theta_den_chol = torch.cholesky(theta_den + 1E-2 * torch.eye(self.rbf_dim))
        del_theta = torch.cholesky_solve(theta_num, theta_den_chol)  # [F, A]

        return del_theta

    def train(self):
        del_theta = self.reweight()
        self.theta = self.theta + del_theta
