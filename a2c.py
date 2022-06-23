import torch
import torch.nn.functional as F
import numpy as np

from torch.distributions import Normal
from replay_buffer import ReplayBuffer

class A2C(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        # hyperparameters
        self.n_step_TD = self.config.hyperparameters['n_step_TD']
        self.crt_h_nodes = self.config.hyperparameters['hidden_nodes']
        self.act_h_nodes = self.config.hyperparameters['hidden_nodes']
        self.crt_learning_rate = self.config.hyperparameters['critic_learning_rate']
        self.act_learning_rate = self.config.hyperparameters['actor_learning_rate']
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT, batch_size=self.n_step_TD)

        self.v_net = self.approximator(self.s_dim, 1, self.crt_h_nodes).to(self.device)
        self.v_net_opt = torch.optim.Adam(self.v_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.a_net = self.approximator(self.s_dim, 2 * self.a_dim, self.act_h_nodes).to(self.device)
        self.a_net_opt = torch.optim.RMSprop(self.a_net.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.trajectory = []

    def ctrl(self, epi, step, x, u, is_train):
        if is_train:
            if epi < self.init_ctrl_idx:
                u_nom = self.initial_ctrl.ctrl(epi, step, x, u)
                u_val = self.explorer.sample(epi, step, u_nom)
            else:
                u_val, _, _ = self.sample_action_and_log_prob(x)
        else:
            _, _, u_val = self.sample_action_and_log_prob(x)

        u_val = np.clip(u_val, -1., 1.)

        return u_val

    def add_experience(self, *single_expr):
        x, u, r, x2, is_term = single_expr
        self.replay_buffer.add(*[x, u, r, x2, is_term])

        if is_term is True: # In on-policy method, clear buffer when episode ends
            self.replay_buffer.clear()

    def sample_action_and_log_prob(self, x):
        """Picks an action using the policy"""
        # Numpy to torch
        x = torch.from_numpy(x.T).float().to(self.device)

        self.a_net.eval()
        with torch.no_grad():
            a_pred = self.a_net(x)
        self.a_net.train()

        u_mean, std = a_pred[:, :self.a_dim], a_pred[:, self.a_dim:]
        std = std.abs()
        u_distribution = Normal(u_mean, std)
        u = u_distribution.sample()
        u_log_prob = u_distribution.log_prob(u) # action은 numpy로 sample 했었음

        # Torch to numpy
        u = u.T.detach().cpu().numpy()
        u_mean = u_mean.T.detach().cpu().numpy()
        return u, u_log_prob, u_mean

    def train(self, step):
        if len(self.replay_buffer) == self.n_step_TD:
            x_traj, u_traj, r_traj, x2_traj, term_traj = self.replay_buffer.sample_sequence()
            _, u_log_prob_traj, _ = self.sample_action_and_log_prob(x_traj.T.detach().cpu().numpy())

            v_target_traj = []

            if term_traj[-1]:  # When Final value of sequence is terminal sample
                v_target_traj.append(r_traj[-1])  # Append terminal cost
            else:  # When Final value of sequence is path sample
                v_target_traj.append(self.v_net(x2_traj[-1]))  # Append n-step bootstrapped q-value

            for i in range(len(self.replay_buffer)):
                v_target_traj.append(r_traj[-i-1] + v_target_traj[-1])

            v_target_traj.reverse()
            v_target_traj = torch.stack(v_target_traj[:-1])
            v_target_traj.detach()

            v_traj = self.v_net(x_traj)
            advantage_traj = v_target_traj - v_traj

            critic_loss = F.mse_loss(v_target_traj, v_traj)
            actor_loss = (u_log_prob_traj * advantage_traj).mean()
            total_loss = critic_loss + actor_loss

            self.v_net_opt.zero_grad()
            self.a_net_opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), self.grad_clip_mag)
            torch.nn.utils.clip_grad_norm_(self.a_net.parameters(), self.grad_clip_mag)
            self.v_net_opt.step()
            self.a_net_opt.step()

            total_loss = total_loss.detach().cpu().item()
        else:
            total_loss = 0.

        return total_loss
