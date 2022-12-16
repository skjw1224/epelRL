import torch
import utils
import numpy as np
import matplotlib.pyplot as plt

from Trajectory_buffer import trajectory_buffer

BUFFER_SIZE = 600
MINIBATCH_SIZE = 32
TAU = 0.05
EPSILON = 0.1
EPI_DENOM = 1.

LEARNING_RATE = 2E-4
ADAM_EPS = 1E-4
L2REG = 1E-3
GRAD_CLIP = 5.0

INITIAL_POLICY_INDEX = 5
Saving_episode = 10
AC_PE_TRAINING_INDEX = 10

class LSTD(object):
    def __init__(self, env, device):
        self.state_dim = env.s_dim
        self.action_dim = env.a_dim
        self.trajectories = trajectory_buffer(env,device,Saving_episode)
        self.feature_dimension = 7
        self.parameter = torch.zeros(self.feature_dimension)
        self.discount = 0.99
        self.action_discrete_number = 5 # Unifying for now
        self.action_mesh = torch.zeros(((self.action_discrete_number, self.action_discrete_number, self.action_dim)))
        for k in range(self.action_discrete_number):
            for kk in range(self.action_discrete_number):
                self.action_mesh[k, kk] = torch.tensor([k - 2, kk - 2])
        self.device = device

        #self.exp_noise = OU_Noise(size=self.env_a_dim)
        # self.initial_ctrl = InitialCntrol(env, device)

        self.Q = env.Q
        self.R = env.R

        """State discrete"""
        # 구간 개수 설정 -> 홀수로 지정
        self.CA_Num = 11
        self.CB_Num = 11
        self.T_Num = 11
        self.VdotVR_Num = 11
        self.CA_class, self.CB_class, self.T_class, self.VdotVR_class = self.setting_discrete_state_index()

        """Action discrete"""
        # 구간 개수 설정 -> 홀수로 지정
        self.dVdotVR_Num = 5
        self.dQKdot_Num = 5
        self.dVdotVR_class, self.dQKdot_class, self.dVdotVR_itv, self.dQdot_itv = self.setting_discrete_action_index()

    def lstd_ctrl(self, epi, step, *single_expr):
        state, action, reward, next_state, term = single_expr
        discrete_state = self.state_to_discrete_state(state)
        discrete_action = self.action_to_discrete_action(action)
        discrete_reward = self.to_discrete_reward(discrete_state,discrete_action)
        self.trajectories.add(np.mod(epi,Saving_episode),discrete_state,discrete_action,discrete_reward)

        if action is None:
            action = self.choose_action(epi, step, discrete_state)

        #single_expr = (state, action, reward, next_state, term)
        #self.replay_buffer.add(*single_expr)
        action_now = self.choose_action(epi, step, discrete_state)
        return action_now

    def lstd_choose_action(self, epi, step, discrete_state):
        if epi < INITIAL_POLICY_INDEX:
            discrete_action = self.initial_policy(discrete_state)
            action = self.discrete_action_to_action(discrete_action)
        else:
            values = torch.tensor(self.action_discrete_number**2)
            for k in range(self.action_discrete_number**2):
                values[k] = (self.eval(self.parameter,discrete_state,self.action_mesh[k//self.action_discrete_number,np.mod(k,self.action_discrete_number)]))
            count = torch.argmax(values)
            action = self.action_mesh[count//self.action_discrete_number, np.mod(count,self.action_discrete_number)]
        return action


    def state_to_discrete_state(self, state):

        x_con = state # Assume size 1 X 7

        CA_d_val, CA_d_idx = torch.min(torch.abs(x_con[1] - self.CA_class).unsqueeze(0), 1)
        CB_d_val, CB_d_idx = torch.min(torch.abs(x_con[2] - self.CB_class).unsqueeze(0), 1)
        T_d_val, T_d_idx = torch.min(torch.abs(x_con[3] - self.T_class).unsqueeze(0), 1)
        VdotVR_d_val, VdotVR_d_idx = torch.min(torch.abs(x_con[5] - self.VdotVR_class).unsqueeze(0), 1)

        CA_dis = CA_d_idx - int(self.CA_Num / 2)
        CB_dis = CB_d_idx - int(self.CB_Num / 2)
        T_dis = T_d_idx - int(self.T_Num / 2)
        VdotVR_dis = VdotVR_d_idx - int(self.VdotVR_Num / 2)

        discrete_state = torch.zeros([7])   # size 1 X 7
        discrete_state[1] = CA_dis
        discrete_state[2] = CB_dis
        discrete_state[3] = T_dis
        discrete_state[5] = VdotVR_dis

        return discrete_state

    def action_to_discrete_action(self, action):

       a_con = action

       dVdotVR_d_val, dVdotVR_d_idx = torch.min(torch.abs(a_con[0] - self.dVdotVR_class).unsqueeze(0), 1)
       dQKdot_d_val, dQKdot_d_idx = torch.min(torch.abs(a_con[1] - self.dQKdot_class).unsqueeze(0), 1)

       dVdotVR_dis = dVdotVR_d_idx - int(self.dVdotVR_Num / 2)
       dQKdot_dis = dQKdot_d_idx - int(self.dQKdot_Num / 2)

       discrete_action = torch.zeros([2])  # size 1 X 7
       discrete_action[0] = dVdotVR_dis
       discrete_action[1] = dQKdot_dis

       return discrete_action

    def to_discrete_reward(self, state, action, data_type):

        x = state
        u = action

        Q = self.Q
        R = self.R

        # y = self.output_functions(x, u)
        # ref = utils.scale(self.ref_traj(), self.ymin, self.ymax)

        if data_type == 'path':
            # cost = 0.5 * torch.chain_matmul(y - ref, Q, (y - ref).T) + 0.5 * torch.chain_matmul(u, R, u.T)
            discrete_reward = 0.5 * torch.chain_matmul(x[2], Q, x[2].T) + 0.5 * torch.chain_matmul(u, R, u.T)
        else:  # terminal condition
            discrete_reward = 0.5 * torch.chain_matmul(x[2], Q, x[2].T)

        return discrete_reward

    def discrete_action_to_action(self, discrete_action):

        a_dis = discrete_action

        dVdotVR_d_to_c = a_dis[0] * self.dVdotVR_itv
        dQKdot_d_to_C = a_dis[1] * self.dQdot_itv

        action = torch.zeros([2])
        action[0] = dVdotVR_d_to_c
        action[1] = dQKdot_d_to_C

        return action

    def setting_discrete_state_index(self):

        CA_ss = 0.7
        CB_ss = 0.05
        T_ss = -0.132
        VdotVR_ss = -0.0216

        CA_itv = 0.1
        CB_itv = 0.1
        T_itv = 0.1
        VdotVR_itv = 0.05

        """홀수로 설정"""
        CA_Num = self.CA_Num
        CB_Num = self.CB_Num
        T_Num = self.T_Num
        VdotVR_Num = self.VdotVR_Num

        CA_class = CA_ss + torch.linspace(-CA_itv * int(CA_Num/2), CA_itv * int(CA_Num/2), CA_Num)
        CB_class = CB_ss + torch.linspace(-CB_itv * int(CB_Num/2), CB_itv * int(CB_Num/2), CB_Num)
        T_class = T_ss + torch.linspace(-T_itv * int(T_Num/2), T_itv * int(T_Num/2), T_Num)
        VdotVR_class = VdotVR_ss + torch.linspace(-VdotVR_itv * int(VdotVR_Num/2), VdotVR_itv * int(VdotVR_Num/2), VdotVR_Num)

        return CA_class, CB_class, T_class, VdotVR_class

    def setting_discrete_action_index(self):

        dVdotVR_ss = 0.0
        dQKdot_ss = 0.0

        dVdotVR_itv = 6
        dQKdot_itv = 6

        """홀수로 설정"""
        dVdotVR_Num = self.dVdotVR_Num
        dQKdot_Num = self.dQKdot_Num

        dVdotVR_class = dVdotVR_ss + torch.linspace(-dVdotVR_itv * int(dVdotVR_Num/2), dVdotVR_itv * int(dVdotVR_Num/2), dVdotVR_Num)
        dQKdot_class = dQKdot_ss + torch.linspace(-dQKdot_itv * int(dQKdot_Num/2), dQKdot_itv * int(dQKdot_Num/2), dQKdot_Num)

        return dVdotVR_class, dQKdot_class, dVdotVR_itv, dQKdot_itv

    def initial_policy(self, discrete_state):
        if discrete_state[5] > 1: k = 0
        elif discrete_state[5] == 1: k = 1
        elif discrete_state[5] == -1: k = 3
        elif discrete_state[5] <  -1: k = 4
        else: k = 2
        if discrete_state[6] > 1: kk = 0
        elif discrete_state[6] == 1: kk = 1
        elif discrete_state[6] == -1: kk = 3
        elif discrete_state[6] < -1: kk = 4
        else: k = 2
        discrete_action = self.action_mesh[k,kk]

        return discrete_action

    def to_feature(self, discrete_state, discrete_action):
        return feature

    def make_discount_sequence(self,length):
        r = torch.zeros((1,length))
        for k in range(length):
            r[0][k] = self.discount**(k)
        return r

    def Train(self, number):
        parameter = torch.zeros((number,self.feature_dimension))
        for p in range(number):
            trajectory = self.trajectories.get_trajectory(p)
            data_length = len(trajectory)/3
            feature_list = torch.zeros((data_length,self.feature_dimension))
            reward_list = torch.zeros((data_length,1))
            reward_discount_list = torch.zeors((data_length,1))
            discount_sequence = self.make_discount_sequence(data_length)
            for q in range(data_length):
                feature_list[q,:] = self.to_feature(trajectory[3*q],trajectory[3*q+1])
                reward_list[q] = trajectory[3*q+2]
            for q in range(data_length):
                reward_discount_list[q] = torch.mm(discount_sequence[:,0:len(discount_sequence[0])-q],reward_list[q:,:])
            C1 = torch.mm(feature_list.transpose(1,2),feature_list)/data_length
            C2 = torch.mm(feature_list[0:len(feature_list)-1,:].transpose(1,2),feature_list[1:len(feature_list),:])/(data_length-1)
            D = torch.mm(feature_list.transpose(1,2),reward_discount_list)/data_length
            C = C1 - self.discount*C2
            parameter[p] = torch.mm(torch.inverse(C),D)
        self.parameter = torch.mean(parameter,0) # using mean

    def eval(self, parameter, discrete_state, discrete_action):
        feature = self.to_feature(discrete_state,discrete_action)
        return torch.dot(parameter,feature)


#############################################################################################################################
    def dqn_ctrl(self, epi, step, *single_expr):
        x, u_idx, r, x2, term = single_expr
        if u_idx is None:
            u_idx, _ = self.choose_action(epi, step, x)

        single_expr = (x, u_idx, r, x2, term)
        self.replay_buffer.add(*single_expr)

        a_idx, a_val = self.choose_action(epi, step, x)
        if epi>= 1:
            self.train()
        return a_idx, a_val

    def choose_action(self, epi, step, x):
        self.q_net.eval()
        with torch.no_grad():
            a_idx = self.q_net(x).min(-1)[1].unsqueeze(1) # (B, A)
        self.q_net.train()

        if step == 0: self.exp_schedule(epi)
        if random.random() <= self.epsilon and epi <= AC_PE_TRAINING_INDEX:
            a_idx = torch.randint(self.a_dim, [1, 1]) # (B, A)
        a_nom = self.action_idx2mesh(vec_idx=a_idx)

        return a_idx, a_nom

    def exp_schedule(self, epi):
        self.epsilon = EPSILON / (1. + (epi / EPI_DENOM))

    def action_idx2mesh(self, vec_idx):
        mesh_idx = (self.a_mesh_idx == vec_idx).nonzero().squeeze(0)
        a_nom = torch.tensor([self.a_mesh[i, :][tuple(mesh_idx)] for i in range(self.env_a_dim)]).float().unsqueeze(0).to(self.device)
        return a_nom

    def train(self):
        s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample()

        q_batch = self.q_net(s_batch).gather(1, a_batch.long())

        q2_batch = self.target_q_net(s2_batch).detach().min(-1)[0].unsqueeze(1) * (~term_batch)
        q_target_batch = r_batch + q2_batch

        q_loss = F.mse_loss(q_batch, q_target_batch)

        self.q_net_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP)
        self.q_net_opt.step()

        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(TAU * from_model.data + (1 - TAU) * to_model.data)

class InitialControl(object):
    def __init__(self, env, device):
        from pid import PID
        self.pid = PID(env, device)

    def controller(self, step, t, x):
        return self.pid.pid_ctrl(step, t, x)
