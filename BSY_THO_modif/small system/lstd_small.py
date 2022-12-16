import torch
import utils
import numpy as np
import matplotlib.pyplot as plt
import random

from Trajectory_buffer import trajectory_buffer

INITIAL_POLICY_INDEX = 50
Saving_episode = 100
Initial_policy_parameter = 0.2
Policy_exploration_parameter = 0.2
learning_rate = 0.1

class LSTD_small(object):
    def __init__(self, env, device):
        self.state_dim = env.s_dim
        self.action_dim = env.a_dim
        self.trajectories = trajectory_buffer(env,device,Saving_episode)
        self.discount = 0.5
        self.action_discrete_number = 5
        self.action_mesh = torch.tensor([-0.3, -0.1, 0.0, 0.1, 0.3])
        self.device = device

        #self.exp_noise = OU_Noise(size=self.env_a_dim)
        # self.initial_ctrl = InitialCntrol(env, device)

        self.Q = env.Q
        self.R = env.R

        """State discrete"""
        # 구간 개수 설정 -> 홀수로 지정
        self.x_Num = 11
        self.x_class = self.setting_discrete_state_index()

        """Action discrete"""
        # 구간 개수 설정 -> 홀수로 지정
        self.u_Num = 5
        self.u_class, self.u_itv = self.setting_discrete_action_index()

        '''Basis discrete'''
        self.feature_parameter = 0.1
        self.feature_dimension = 3  # Should change!
        self.parameter = torch.ones(self.feature_dimension)
        self.Feature_point_candidate = []
        self.Feature_point_candidate.append([0.])
        self.Feature_point_candidate.append([0.])

    def ctrl(self, epi, step, *single_expr):
        state, action, reward, next_state, is_term, _ = single_expr
        state = state[0]
        action = action[0]
        reward = reward[0]
        next_state = next_state[0]
        is_term = is_term[0][0]
        discrete_state = self.state_to_discrete_state(state)
        print("discrete state:", discrete_state)
        discrete_state_next = self.state_to_discrete_state(next_state)
        discrete_action = self.action_to_discrete_action(action)
        discrete_reward = self.to_discrete_reward(discrete_state, discrete_action, is_term)
        #print('state,reward :', discrete_state, discrete_reward)
        self.trajectories.add(np.mod(epi, Saving_episode), discrete_state, discrete_action, discrete_reward)

        # if action is None:
        #     action = self.choose_action(epi, step, discrete_state)

        #single_expr = (state, action, reward, next_state, term)
        #self.replay_buffer.add(*single_expr)
        action_now = self.lstd_choose_action(epi, step, discrete_state_next)

        if is_term:
            if epi > 0:
                if np.mod(epi,Saving_episode) == 0: self.Train(Saving_episode)
        # return action_now.view(1,-1)
        return action_now

    def lstd_choose_action(self, epi, step, discrete_state):
        if epi < INITIAL_POLICY_INDEX:
            action = self.initial_policy(discrete_state).unsqueeze(0)
            # discrete_action = self.initial_policy(discrete_state)
            # action = self.discrete_action_to_action(discrete_action)
        else:
            values = torch.zeros(self.action_discrete_number)
            for k in range(self.action_discrete_number):
                values[k] = self.eval(self.parameter, discrete_state, self.action_mesh[k])
            count = torch.argmax(values)
            print('discrete state:', discrete_state)
            print("value:", values, "count: ", count)
            action = self.action_mesh[count]
            action = torch.tensor([action])
        print("Action value:", action)
        # if random.random() < (Policy_exploration_parameter - 0*epi):
        #     action = self.action_mesh[random.randrange(0,self.action_discrete_number), random.randrange(0,self.action_discrete_number)]
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
        x_con = state # Assume size 1 X 1
        x_d_val, x_d_idx = torch.min(torch.abs(x_con[0] - self.x_class).unsqueeze(0), 1)
        x_dis = x_d_idx - int(self.x_Num / 2)
        discrete_state = torch.zeros([1])  # size 1 X 7
        discrete_state[0] = x_dis
        return discrete_state

    def action_to_discrete_action(self, action):
        if action == -0.3:
            discrete_action = 0
        elif action == -0.1:
            discrete_action = 1
        elif action == 0.1:
            discrete_action = 3
        elif action == 0.3:
            discrete_action = 4
        else:
            discrete_action = 2
        return discrete_action

    def to_discrete_reward(self, state, action, is_term):
        u = torch.ones((1,self.action_dim))
        # u = action
        u[0] = action
        # x = torch.diag(torch.tensor([state[2]]))
        x = torch.diag(torch.tensor([state]))
        # x = state

        Q = self.Q
        R = self.R

        # y = self.output_functions(x, u)
        # ref = utils.scale(self.ref_traj(), self.ymin, self.ymax)

        if is_term:
            # cost = 0.5 * torch.chain_matmul(y - ref, Q, (y - ref).T) + 0.5 * torch.chain_matmul(u, R, u.T)
            discrete_reward = (50 - 0.5 * torch.chain_matmul(x, Q, x.T) - 0.5 * torch.chain_matmul(u, R, u.T))
        else:  # terminal condition
            discrete_reward = (50 - 0.5 * torch.chain_matmul(x, Q, x.T))
        return discrete_reward[0]

    # def discrete_action_to_action(self, discrete_action):
    #     a_dis = discrete_action
    #
    #     dVdotVR_d_to_c = a_dis[0] * self.dVdotVR_itv
    #     dQKdot_d_to_C = a_dis[1] * self.dQdot_itv
    #
    #     action = torch.zeros([2])
    #     action[0] = dVdotVR_d_to_c
    #     action[1] = dQKdot_d_to_C
    #
    #     return action

    def setting_discrete_state_index(self):
        ''' should be in init'''
        x_ss = 0
        x_itv = 0.2

        """홀수로 설정"""
        x_Num = self.x_Num

        x_class = x_ss + torch.linspace(-x_itv * int(x_Num / 2), x_itv * int(x_Num / 2), x_Num)

        return x_class

    def setting_discrete_action_index(self):
        '''should be in init / 우ㅠ우different with original code'''
        u_ss = -0.3
        u_itv = 0.2

        """홀수로 설정"""
        # u_Num = self.u_Num

        u_class = u_ss + u_itv * torch.arange(0, 5)

        return u_class, u_itv

    def initial_policy(self, discrete_state):
        if discrete_state > 1:
            k = 0
        elif discrete_state == 1:
            k = 1
        elif discrete_state == -1:
            k = 3
        elif discrete_state < -1:
            k = 4
        else:
            k = 2
        discrete_action = self.action_mesh[k]   # action_mesh = [-0.3, -0.1, 0.0, 0.1, 0.3]
        return discrete_action

    def to_feature(self, discrete_state, discrete_action):
        state_and_action = torch.zeros(self.state_dim + self.action_dim)
        state_and_action[0:self.state_dim] = discrete_state
        state_and_action[self.state_dim:] = discrete_action
        feature = torch.zeros(self.feature_dimension)
        for k in range(self.feature_dimension-1):
            feature[k] = state_and_action[k]**2
        feature[self.feature_dimension-1] = 1.
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
        feature = self.to_feature(discrete_state, discrete_action)
        return torch.dot(parameter, feature)

    '''
       def lstd_choose_action(self, epi, step, discrete_state):
            if epi < INITIAL_POLICY_INDEX:
                discrete_action = self.initial_policy(discrete_state)
                action = self.discrete_action_to_action(discrete_action)
            else:
                values = torch.zeros(self.action_discrete_number**2)
                for k in range(self.action_discrete_number**2):
                    values[k] = self.eval(self.parameter, discrete_state, self.action_mesh[k//self.action_discrete_number, np.mod(k,self.action_discrete_number)])
                count = torch.argmax(values)
                print('state',discrete_state)
                print(values, count)
                action = self.action_mesh[count//self.action_discrete_number, np.mod(count,self.action_discrete_number)]
            if random.random() < (Policy_exploration_parameter - 0*epi):
                action = self.action_mesh[random.randrange(0,self.action_discrete_number), random.randrange(0,self.action_discrete_number)]
        return action
        
        
        
        
        
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