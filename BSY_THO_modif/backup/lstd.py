import torch
import utils
import numpy as np
import matplotlib.pyplot as plt
import random

from Trajectory_buffer import trajectory_buffer

INITIAL_POLICY_INDEX = 100
Saving_episode = 50
Initial_policy_parameter = 0.4
Policy_exploration_parameter = 0.2
learning_rate = 0.1

class LSTD(object):
    def __init__(self, env, device):
        self.state_dim = env.s_dim
        self.action_dim = env.a_dim
        self.trajectories = trajectory_buffer(env,device,Saving_episode)
        self.discount = 0.5
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
        self.CB_Num = 21
        self.T_Num = 11
        self.VdotVR_Num = 11 # Vdot -> V change require
        self.Qdot_Num = 11
        self.CA_class, self.CB_class, self.T_class, self.VdotVR_class, self.Qdot_class = self.setting_discrete_state_index()

        """Action discrete"""
        # 구간 개수 설정 -> 홀수로 지정
        self.dVdotVR_Num = 5
        self.dQKdot_Num = 5
        self.dVdotVR_class, self.dQKdot_class, self.dVdotVR_itv, self.dQdot_itv = self.setting_discrete_action_index()

        '''Basis discrete'''
        self.feature_parameter = 0.1
        self.feature_dimension = 19  # Should change!
        self.parameter = torch.ones(self.feature_dimension)
        self.Feature_point_candidate = []
        self.Feature_point_candidate.append([0.])
        self.Feature_point_candidate.append([0.])
        self.Feature_point_candidate.append([0.])
        self.Feature_point_candidate.append([0.])
        self.Feature_point_candidate.append([0.])
        self.Feature_point_candidate.append([0.])
        self.Feature_point_candidate.append([0.])
        self.Feature_point_candidate.append([0.])
        self.Feature_point_candidate.append([0.])

    def ctrl(self, epi, step, *single_expr):
        state, action, reward, next_state, is_term = single_expr
        state = state[0]
        action = action[0]
        reward = reward[0]
        next_state = next_state[0]
        is_term = is_term[0][0]
        discrete_state = self.state_to_discrete_state(state)
        discrete_next_state = self.state_to_discrete_state(next_state)
        discrete_action = self.action_to_discrete_action(action)
        discrete_reward = self.to_discrete_reward(discrete_state,discrete_action,is_term)
        #print('state,reward :', discrete_state, discrete_reward)
        self.trajectories.add(np.mod(epi,Saving_episode),discrete_state,discrete_action,discrete_reward)

        if action is None:
            action = self.choose_action(epi, step, discrete_next_state)

        #single_expr = (state, action, reward, next_state, term)
        #self.replay_buffer.add(*single_expr)
        action_now = self.lstd_choose_action(epi, step, discrete_next_state)

        if is_term:
            if epi > 0:
                if np.mod(epi,Saving_episode) == 0: self.Train(Saving_episode)
        return action_now.view(1,-1)

    def lstd_choose_action(self, epi, step, discrete_state):
        if epi < INITIAL_POLICY_INDEX:
            discrete_action = self.initial_policy(discrete_state)
            action = self.discrete_action_to_action(discrete_action)
        else:
            values = torch.zeros(self.action_discrete_number**2)
            for k in range(self.action_discrete_number**2):
                values[k] = self.eval(self.parameter,discrete_state,self.action_mesh[k//self.action_discrete_number,np.mod(k,self.action_discrete_number)])
            count = torch.argmax(values)
            print('state',discrete_state)
            print(values, count)
            action = self.action_mesh[count//self.action_discrete_number, np.mod(count,self.action_discrete_number)]
        if random.random() < (Policy_exploration_parameter - 0*epi):
            action = self.action_mesh[random.randrange(0,self.action_discrete_number), random.randrange(0,self.action_discrete_number)]
        return action

    def indexing(self, i):
        num = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # should be change with feature_point_candidate
        m = []
        for k in range(len(self.Feature_point_candidate)):
            if len(self.Feature_point_candidate[k]) == 1:
                m.append(0.)
            else:
                number = i // num[k]
                m.append(self.Feature_point_candidate[k][number])
                i = i - num[k] * number
        return torch.tensor(m)

    def state_to_discrete_state(self, state):

        x_con = state # Assume size 1 X 7

        CA_d_val, CA_d_idx = torch.min(torch.abs(x_con[1] - self.CA_class).unsqueeze(0), 1)
        CB_d_val, CB_d_idx = torch.min(torch.abs(x_con[2] - self.CB_class).unsqueeze(0), 1)
        T_d_val, T_d_idx = torch.min(torch.abs(x_con[3] - self.T_class).unsqueeze(0), 1)
        VdotVR_d_val, VdotVR_d_idx = torch.min(torch.abs(x_con[5] - self.VdotVR_class).unsqueeze(0), 1)
        Qdot_d_val, Qdot_d_idx = torch.min(torch.abs(x_con[6] - self.Qdot_class).unsqueeze(0), 1)

        CA_dis = CA_d_idx - int(self.CA_Num / 2)
        CB_dis = CB_d_idx - int(self.CB_Num / 2)
        T_dis = T_d_idx - int(self.T_Num / 2)
        VdotVR_dis = VdotVR_d_idx - int(self.VdotVR_Num / 2)
        Qdot_dis = Qdot_d_idx - int(self.Qdot_Num / 2)

        discrete_state = torch.zeros([7])   # size 1 X 7
        discrete_state[1] = CA_dis
        discrete_state[2] = CB_dis
        discrete_state[3] = T_dis
        discrete_state[5] = VdotVR_dis
        discrete_state[6] = Qdot_dis

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

    def to_discrete_reward(self, state, action, is_term):

        u = torch.ones((1,self.action_dim))
        u[0] = action
        x = torch.diag(torch.tensor([state[2]]))
        Q = self.Q
        R = self.R

        # y = self.output_functions(x, u)
        # ref = utils.scale(self.ref_traj(), self.ymin, self.ymax)

        if is_term:
            # cost = 0.5 * torch.chain_matmul(y - ref, Q, (y - ref).T) + 0.5 * torch.chain_matmul(u, R, u.T)
            discrete_reward = (200 - 0.5 * torch.chain_matmul(x, Q, x.T) - 0.5 * torch.chain_matmul(u, R, u.T))
        else:  # terminal condition
            discrete_reward = (200 - 0.5 * torch.chain_matmul(x, Q, x.T))
        return discrete_reward[0]


    def discrete_action_to_action(self, discrete_action):

        a_dis = discrete_action

        dVdotVR_d_to_c = a_dis[0] * self.dVdotVR_itv
        dQKdot_d_to_C = a_dis[1] * self.dQdot_itv

        action = torch.zeros([2])
        action[0] = dVdotVR_d_to_c
        action[1] = dQKdot_d_to_C

        return action

    def setting_discrete_state_index(self):
        ''' should be in init'''
        CA_ss = 0.7
        CB_ss = 0.05
        T_ss = -0.132
        VdotVR_ss = -0.0216
        Qdot_ss = -0.03087

        CA_itv = 0.1
        CB_itv = 0.01
        T_itv = 0.1
        VdotVR_itv = 0.05
        Qdot_itv = 0.2

        """홀수로 설정"""
        CA_Num = self.CA_Num
        CB_Num = self.CB_Num
        T_Num = self.T_Num
        VdotVR_Num = self.VdotVR_Num
        Qdot_Num = self.Qdot_Num

        CA_class = CA_ss + torch.linspace(-CA_itv * int(CA_Num/2), CA_itv * int(CA_Num/2), CA_Num)
        CB_class = CB_ss + torch.linspace(-CB_itv * int(CB_Num/2), CB_itv * int(CB_Num/2), CB_Num)
        T_class = T_ss + torch.linspace(-T_itv * int(T_Num/2), T_itv * int(T_Num/2), T_Num)
        VdotVR_class = VdotVR_ss + torch.linspace(-VdotVR_itv * int(VdotVR_Num/2), VdotVR_itv * int(VdotVR_Num/2), VdotVR_Num)
        Qdot_class = Qdot_ss + torch.linspace(-Qdot_itv * int(Qdot_Num / 2), Qdot_itv * int(Qdot_Num / 2), Qdot_Num)

        return CA_class, CB_class, T_class, VdotVR_class, Qdot_class

    def setting_discrete_action_index(self):
        '''should be in init'''
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
        else: kk = 2
        discrete_action = self.action_mesh[k,kk]
        if random.random() < Initial_policy_parameter:
            discrete_action = self.action_mesh[random.randrange(0,self.action_discrete_number), random.randrange(0,self.action_discrete_number)]
        return discrete_action

    def to_feature(self, discrete_state, discrete_action):
        state_and_action = torch.zeros(self.state_dim + self.action_dim)
        state_and_action[0:self.state_dim] = discrete_state
        state_and_action[self.state_dim:] = discrete_action
        feature = torch.zeros(self.feature_dimension)
        for k in range(int((self.feature_dimension-1)/2)):
            feature[k] = state_and_action[k]**2
            feature[int((self.feature_dimension-1)/2) + k] = state_and_action[k]
        feature[self.feature_dimension-1] = 1.
        return feature


    '''
    def to_feature(self, discrete_state, discrete_action):
        state_and_action = torch.zeros(self.state_dim+self.action_dim)
        state_and_action[0:self.state_dim] = discrete_state
        state_and_action[self.state_dim:] = discrete_action
        feature = torch.zeros(self.feature_dimension)
        for k in range(self.feature_dimension):
            feature[k] = torch.exp(-(self.feature_parameter*torch.norm(state_and_action - self.indexing(k))))
        return feature
    '''
    def make_discount_sequence(self,length):
        r = torch.zeros((1,length))
        for k in range(length):
            r[0][k] = self.discount**(k)
        return r

    def Train(self, number):
        parameter = torch.zeros((number,self.feature_dimension))
        for p in range(number):
            trajectory = self.trajectories.get_trajectory(p)
            data_length = len(trajectory)//3
            feature_list = torch.zeros((data_length,self.feature_dimension))
            reward_list = torch.zeros((data_length,1))
            reward_discount_list = torch.zeros((data_length,1))
            discount_sequence = self.make_discount_sequence(data_length)
            for q in range(data_length):
                feature_list[q,:] = self.to_feature(trajectory[3*q],trajectory[3*q+1])
                reward_list[q] = trajectory[3*q+2]
            for q in range(data_length):
                reward_discount_list[q] = torch.mm(discount_sequence[:,0:len(discount_sequence[0])-q],reward_list[q:,:])
            C1 = torch.mm(feature_list.transpose(0,1),feature_list)/data_length
            C2 = torch.mm(feature_list[0:len(feature_list)-1,:].transpose(0,1),feature_list[1:len(feature_list),:])/(data_length-1)
            D = torch.mm(feature_list.transpose(0,1),reward_discount_list)/data_length
            C = C1 - self.discount*C2 + torch.diag(10**(-5)*torch.ones(self.feature_dimension)) # Preventing singular matrix
            parameter[p] = torch.mm(torch.inverse(C),D).squeeze()
        self.parameter = self.parameter + learning_rate*torch.mean(parameter,0) # using mean
        print(self.parameter)

    def eval(self, parameter, discrete_state, discrete_action):
        feature = self.to_feature(discrete_state,discrete_action)
        return torch.dot(parameter,feature)

    '''
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
    '''

#############
'''
from env import CstrEnv
device = 'cpu'

env = CstrEnv(device)
b = LSTD(env,device)
for kk in range(30):
    x, y, u, data_type = env.reset()
    for k in range(100):
        x2, y2, u, r, is_term, derivs = env.step(x, u)
        u = b.ctrl(0, 0, x, u, r, x2, is_term)
        x = x2
    b.Train(1)
'''