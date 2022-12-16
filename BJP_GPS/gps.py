import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from cilqr import Ilqr # 추후에 cilqr로 변경할것
from ilqr2 import Ilqr as Ilqr2  # 추후에 cilqr로 변경할것
from replay_buffer import ReplayBuffer

import sys
import os
import matplotlib.pyplot as plt
my_CSTR = os.getcwd()
sys.path.append(my_CSTR)
import gc
from explorers import OU_Noise

MAX_KL = 0.01
BUFFER_SIZE = 1800
MINIBATCH_SIZE = 32
LEARNING_RATE = 0.001#0.0003
ADAM_EPS = 1E-4
L2REG = 0#1E-3

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a deep Q-learning network
        Arguments:
            input_dim : number of state
            output_dim : number of action
        """
        super(PolicyNetwork, self).__init__()
        n_h_nodes = [10, 10]

        self.fc1 = nn.Linear(input_dim, n_h_nodes[0])
        self.fc2 = nn.Linear(n_h_nodes[0], n_h_nodes[1])
        self.fc3 = nn.Linear(n_h_nodes[1], output_dim)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

        # self.log_std = nn.Parameter(torch.zeros(1, output_dim))
        self.elu=torch.nn.ELU()
    def forward(self, x):
        # x = torch.FloatTensor(x)

        # x = torch.tanh(self.fc1(x))
        # x = torch.tanh(self.fc2(x))

        # x = self.elu(self.fc1(x))
        # x = self.elu(self.fc2(x))

        x = torch.relu(self.fc1(x))
        # x = torch.reu(self.fc2(x))

        x = self.fc3(x)
        # logstd = torch.zeros_like(mu)
        # logstd = self.log_std.expand_as(x)
        # std = torch.exp(logstd)
        return x #, std, logstd

class GPS(object):
    def __init__(self, env, device):
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.device = device
        self.nT=env.nT
        ## learning 에 필요한 설정
        self.replay_buffer = ReplayBuffer(env, device, buffer_size=BUFFER_SIZE, batch_size=MINIBATCH_SIZE)
        # self.exp_noise = OU_Noise(self.a_dim)
        # self.initial_ctrl = InitialControl(env, device)

        ## Policy (+old) net
        self.p_net = PolicyNetwork(self.s_dim,self.a_dim).to(device)
        self.old_p_net = PolicyNetwork(self.s_dim, self.a_dim).to(device)
        self.p_net_opt = torch.optim.Adam(self.p_net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS,
                                                    weight_decay=L2REG)
        self.criterion = torch.nn.MSELoss()
        ## Value net (learning 참고용)
        # self.v_net = ValueNetwork(self.s_dim).to(device)
        # self.v_net_opt = torch.optim.Adam(self.v_net.parameters(), lr=CRT_LEARNING_RATE, eps=ADAM_EPS, weight_decay=L2REG)

        # 여러 initial condition에 대해 학습할 경우
        # self.n_cond = env.n_cond # initial condition 개수


        ## local linear gaussian controller 초기값 생성
        # self.lqr=[]
        # self.lqr2 = []
        # for i in range(self.n_cond):
        #     self.lqr.append(Ilqr(env, device))
        #     self.lqr2.append(Ilqr2(env, device))
        self.lqr=Ilqr(env, device)
        self.lqr2 = Ilqr2(env, device) # sympy 기반이라 조금더 빠름
        ## 학습 초기에 ilqr을 먼저 학습하는 에피소드 개수
        self.epi_ilqr = 16 # ilqr 결과가 학습되는 것을 확인할때 까지 증가
        ## gps 학습시 neural net update 횟수
        self.max_it=20000
    def s_step(self,kl_div):
        loss = self.criterion(kl_div, kl_div * 0)
        self.p_net_opt.zero_grad()
        loss.backward(retain_graph=True)
        self.p_net_opt.step() #lkj

    def ctrl(self, epi, step, x, u):
        if epi < self.epi_ilqr: # 학습이 되지 않은 경우
            u = self.lqr2.lqr_ref(step, x, u)
        else: # gps 로 돌려보고 ilqr과 비슷하게 학습하는 과정
            if step==0:
                klist=self.lqr2.gains
                torch.save(klist, my_CSTR + '/data/gps/ilqr_for_gps/klist.pt')

            l_k, L_k, xd, ud = self.lqr2.gains[step]  # constant, gain, 선형화 지점 불러옴
            delx = x - xd
            u = ud + (l_k + torch.matmul(L_k, delx.T)).T
            u_g = self.p_net(x)
            self.replay_buffer.add(*[x, u_g, u, torch.tensor([[step]]), delx])
            if step == self.nT-1: # gps s_step
                u_tensor = 0*u
                for it in range(self.max_it):
                    x_traj, u_g_traj, u_traj, step_traj, delx_traj = self.replay_buffer.sample()
                    kl_div = torch.tensor([[0.]])
                    for k in range(self.replay_buffer.batch_size):
                        x_sample=x_traj[k]
                        step_sample = step_traj[k]
                        u_g=self.p_net(x_sample)

                        l_k, L_k, xd, ud = self.lqr2.gains[step_sample]  # constant, gain, 선형화 지점 불러옴
                        delx = x_sample - xd
                        u = ud + (l_k + torch.matmul(L_k, delx.T)).T
                        kl_div = kl_div + torch.chain_matmul((u - u_g), (u - u_g).T)
                    self.s_step(kl_div)
                    if it % 100 == 0:
                        print(it,'th gps update')
                    if it%1000 == 0:
                        x_traj, u_g_traj, u_traj, step_traj, delx_traj = self.replay_buffer.sample_all()
                        kl_div = torch.tensor([[0.]])
                        u_all = torch.zeros([1, self.a_dim])
                        ug_all = torch.zeros([1, self.a_dim])
                        for i in range(self.nT):
                            x_sample = x_traj[i]
                            step_sample = step_traj[i]
                            u_g = self.p_net(x_sample)+u_tensor

                            l_k, L_k, xd, ud = self.lqr2.gains[step_sample]  # constant, gain, 선형화 지점 불러옴
                            delx = x_sample - xd
                            u = ud + (l_k + torch.matmul(L_k, delx.T)).T+u_tensor
                            u_all = torch.cat((u_all, u))
                            ug_all = torch.cat((ug_all, u_g))
                            kl_div = kl_div + torch.chain_matmul((u - u_g), (u - u_g).T)
                        print(it,'th traj kl_div: ', kl_div)
                        ug_all = ug_all[1:].detach().numpy()
                        u_all = u_all[1:].detach().numpy()

                        plt.rc('xtick', labelsize=20)
                        plt.rc('ytick', labelsize=20)
                        fig = plt.figure(figsize=[20, 12])
                        fig.subplots_adjust(hspace=.4, wspace=.5)
                        label = [r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$']
                        for j in range(len(label)):
                            ax = fig.add_subplot(2, 2, j + 1)
                            ax.save(u_all[:, j])
                            plt.ylabel(label[j], size=30)
                            if j == 0:
                                ax.set_ylim([-0.1, 1.75])
                            else:
                                ax.set_ylim([-0.6, 0.2])
                        for j in range(len(label)):
                            ax = fig.add_subplot(2, 2, j + 3)
                            ax.save(ug_all[:, j])
                            plt.ylabel(label[j], size=30)
                            if j == 0:
                                ax.set_ylim([-0.1, 1.75])
                            else:
                                ax.set_ylim([-0.6, 0.2])
                        plt.savefig(my_CSTR + '/data/gps/ilqr_vs_gps' + str(it) + '.png')
                        plt.clf()
                        plt.close()
                        gc.collect()
        return u
