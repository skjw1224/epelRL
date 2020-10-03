import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#from ou_noise import OU_Noise
from torch.distributions import Normal

from replay_buffer import ReplayBuffer


INITIAL_POLICY_INDEX = 1
AC_PE_TRAINING_INDEX = 950

CRITIC_LEARNING_RATE = 0.0002
ACTOR_LEARNING_RATE = 0.0001
BOOTSTRAP_LENGTH = 10

class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a deep Q-learning network
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(Network, self).__init__()

        n_h_nodes = [50, 50, 30]

        self.fc1 = nn.Linear(input_dim, n_h_nodes[0])
        self.bn1 = nn.BatchNorm1d(n_h_nodes[0])
        self.fc2 = nn.Linear(n_h_nodes[0], n_h_nodes[1])
        self.bn2 = nn.BatchNorm1d(n_h_nodes[1])
        self.fc3 = nn.Linear(n_h_nodes[1], n_h_nodes[2])
        self.bn3 = nn.BatchNorm1d(n_h_nodes[2])
        self.fc4 = nn.Linear(n_h_nodes[2], output_dim)

    def forward(self, x):
        x = F.leaky_relu_(self.fc1(x))
        x = F.leaky_relu_(self.fc2(x))
        x = F.leaky_relu_(self.fc3(x))
        x = self.fc4(x)

        # x = F.leaky_relu(self.bn1(self.fc1(x)))
        # x = F.leaky_relu(self.bn2(self.fc2(x)))
        # x = F.leaky_relu(self.bn3(self.fc3(x)))
        # x = F.leaky_relu(self.fc4(x))
        return x


class Actor_Critic(object):
    def __init__(self, env, device):
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.env_epi_length = env.nT

        self.device = device

        self.replay_buffer = ReplayBuffer(env, device, buffer_size=self.env_epi_length, batch_size=BOOTSTRAP_LENGTH)
        self.initial_ctrl = InitialControl(env, device)

        self.v_net = Network(self.s_dim, 1).to(device)
        self.v_net_opt = optim.Adam(self.v_net.parameters(), lr=CRITIC_LEARNING_RATE)

        self.a_net = Network(self.s_dim, 2 * self.a_dim).to(device)
        self.a_net_opt = optim.RMSprop(self.a_net.parameters(), lr=ACTOR_LEARNING_RATE)

        self.Trajectory = []

        #self.exp_noise = OU_Noise(size=self.action_dimension)

    def ctrl(self, epi, step, x, u):
        if epi < INITIAL_POLICY_INDEX:
            a_nom = self.initial_ctrl.controller(step, x, u)
            a_nom.detach()
            a_exp = self.exp_schedule(epi, step)
            a_val = a_nom + a_exp
        elif INITIAL_POLICY_INDEX <= epi < AC_PE_TRAINING_INDEX:
            a_val, _, _ = self.sample_action_and_log_prob(x)
            a_val.squeeze(0)
        else:
            _, _, a_val = self.sample_action_and_log_prob(x)
            a_val.squeeze(0)

        a_val = torch.clamp(a_val, -1., 1.)

        if len(self.replay_buffer) == BOOTSTRAP_LENGTH:
            self.nn_train()

        return a_val

    def add_experience(self, *single_expr):
        x, u, r, x2, term = single_expr
        self.replay_buffer.add(*[x, u, r, x2, term])

        if term is True: # In on-policy method, clear buffer when episode ends
            self.replay_buffer.clear()

    def exp_schedule(self, epi, step):
        # a_val += (1 - epi / 1000) * torch.tensor([np.random.normal(0, 0.01, self.a_dim)])
        a_exp = 0.99 ** epi * torch.normal(0, 1, size=[1, self.a_dim])
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

    def nn_train(self):
        s_traj, a_traj, r_traj, v_target_traj, a_log_prob_traj = self.eval_traj_return_a_log_prob()
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


    def eval_traj_return_a_log_prob(self):
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

    def controller(self, step, x, u):
        return self.pid.ctrl(step, x)
