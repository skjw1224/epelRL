import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#from ou_noise import OU_Noise
from torch.distributions import Normal
from nn_create import NeuralNetworks

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
        self.bootstrap_length = self.config.hyperparameters['bootstrap_length']
        self.crt_learning_rate = self.config.hyperparameters['critic_learning_rate']
        self.act_learning_rate = self.config.hyperparameters['actor_learning_rate']
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.explore_epi_idx = self.config.hyperparameters['explore_epi_idx']
        self.eps_decay_rate = self.config.hyperparameters['eps_decay_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT, batch_size=self.bootstrap_length)
        self.initial_ctrl = InitialControl(self.env, self.device)

        self.v_net = NeuralNetworks(self.s_dim, 1).to(self.device)
        self.v_net_opt = optim.Adam(self.v_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.a_net = NeuralNetworks(self.s_dim, 2 * self.a_dim).to(self.device)
        self.a_net_opt = optim.RMSprop(self.a_net.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.trajectory = []

        #self.exp_noise = OU_Noise(size=self.action_dimension)

    def ctrl(self, epi, step, x, u):
        if epi < self.init_ctrl_idx:
            u_nom = self.initial_ctrl.controller(epi, step, x, u)
            u_val = u_nom + self.exp_schedule(epi, step)
        elif self.init_ctrl_idx <= epi < self.explore_epi_idx:
            u_val, _, _ = self.sample_action_and_log_prob(x)
            u_val.squeeze(0)
        else:
            _, _, u_val = self.sample_action_and_log_prob(x)
            u_val.squeeze(0)

        u_val = np.clip(u_val, -1., 1.)

        return u_val

    def add_experience(self, *single_expr):
        x, u, r, x2, is_term = single_expr
        self.replay_buffer.add(*[x, u, r, x2, is_term])

        if is_term is True: # In on-policy method, clear buffer when episode ends
            self.replay_buffer.clear()

    def exp_schedule(self, epi, step):
        # a_val += (1 - epi / 1000) * torch.tensor([np.random.normal(0, 0.01, self.a_dim)])
        a_exp = self.eps_decay_rate ** epi * torch.normal(0, 1, size=[1, self.a_dim])
        return a_exp

    def sample_action_and_log_prob(self, x):
        """Picks an action using the policy"""
        #state = torch.from_numpy(state).float().unsqueeze(0) # tensor form으로 들어가나?

        self.a_net.eval()
        with torch.no_grad():
            a_pred = self.a_net(x)
        self.a_net.train()

        mean, std = a_pred[:, :self.a_dim], a_pred[:, self.a_dim:]
        std = std.abs()
        action_distribution = Normal(mean, std)
        action = action_distribution.sample()
        action_log_prob = action_distribution.log_prob(action) # action은 numpy로 sample 했었음
        return action, action_log_prob, mean

    def train(self, step):
        if len(self.replay_buffer) == self.bootstrap_length:
            s_traj, a_traj, r_traj, v_target_traj, a_log_prob_traj = self.eval_traj_r_a_log_prob()
            #print(traject)
            #print(Traject_state[0].grad_fn)
            #print(Traject_action_log_prob)
            #print(R)

            total_loss = self.eval_total_loss(v_target_traj, s_traj, a_log_prob_traj)

            # torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), self.gradient_clipping_norm) #clipping??
            self.v_net.zero_grad()
            self.a_net.zero_grad()
            total_loss.backward()
            self.v_net_opt.step()
            self.a_net_opt.step()


    def eval_traj_r_a_log_prob(self):
        # Replay on-policy trajectory
        s_traj, a_traj, r_traj, s2_traj, term_traj = self.replay_buffer.sample_sequence()
        _, a_log_prob_traj, _ = self.sample_action_and_log_prob(s_traj)

        # Calculate cumulative return (q value estimates) through sampled trajectory
        v_target_traj = []

        if term_traj[-1]: # When Final value of sequence is terminal sample
            v_target_traj.append(r_traj[-1])  # Append terminal cost
        else: # When Final value of sequence is path sample
            v_target_traj.append(self.v_net(s2_traj[-1])) # Append n-step bootstrapped q-value

        for i in range(len(self.replay_buffer)):
            v_target_traj.append(r_traj[-i-1] + v_target_traj[-1])

        v_target_traj.reverse()
        v_target_traj = torch.stack(v_target_traj[:-1]) # (B, 1)

        return s_traj, a_traj, r_traj, v_target_traj, a_log_prob_traj

    def eval_total_loss(self, v_target_traj, s_traj, a_log_prob_traj):
        v_traj = self.v_net(s_traj)
        advantage_traj = v_target_traj - v_traj
        advantage_traj = advantage_traj.detach()
        critic_loss = F.mse_loss(v_traj, v_target_traj)

        actor_loss = a_log_prob_traj * advantage_traj  ## -1 부호 확인, cat size 확인
        actor_loss = actor_loss.mean()
        total_loss = critic_loss + actor_loss
        return total_loss

from pid import PID
class InitialControl(object):
    def __init__(self, env, device):
        self.pid = PID(env, device)

    def controller(self, epi, step, x, u):
        return self.pid.ctrl(epi, step, x, u)
