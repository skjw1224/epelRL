import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ilqr import iLQR # 추후에 cilqr로 변경할것
from replay_buffer import ReplayBuffer


class GPS(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        # hyperparameters
        self.h_nodes = self.config.hyperparameters['hidden_nodes']
        self.buffer_size = self.config.hyperparameters['buffer_size']
        self.minibatch_size = self.config.hyperparameters['minibatch_size']
        self.learning_rate = self.config.hyperparameters['learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.ilqr_episode = self.config.hyperparameters['ilqr_episode']

        self.approximator = self.config.algorithm['approximator']['function']

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.minibatch_size)

        # Policy network
        self.a_net = self.approximator(self.s_dim, self.a_dim, self.h_nodes).to(self.device)
        self.a_net_opt = optim.Adam(self.a_net.parameters(), lr=self.learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        ## Value net (learning 참고용)
        # self.v_net = ValueNetwork(self.s_dim).to(device)
        # self.v_net_opt = torch.optim.Adam(self.v_net.parameters(), lr=CRT_LEARNING_RATE, eps=ADAM_EPS, weight_decay=L2REG)

        ## local linear gaussian controller 초기값 생성
        # self.lqr=[]
        # self.lqr2 = []
        # for i in range(self.n_cond):
        #     self.lqr.append(Ilqr(env, device))
        #     self.lqr2.append(Ilqr2(env, device))
        self.lqr = ILQR(env, device)

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
                u_tensor=0*u
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

        return u

    def train(self):
        None