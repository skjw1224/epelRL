import torch
import numpy as np
import utils

import time

class Ilqr(object):
    def __init__(self, env, device): # environment의 정보들을 가져온다
        self.env = env
        self.device = device
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.p_dim = env.p_dim

        self.t0 = env.t0  # ex) 0
        self.tT = env.tT  # ex) 2
        self.nT = env.nT
        self.dt = env.dt  # ex) dt:0.005

        self.dx_eval = env.dx_eval
        self.y_eval = env.y_eval
        self.c_eval = env.c_eval
        self.cT_eval = env.cT_eval
        self.dx_derivs = env.dx_derivs
        self.c_derivs = env.c_derivs
        self.cT_derivs = env.cT_derivs

        self.p_mu, self.p_sigma, self.p_eps = env.param_real, \
                               torch.zeros([1, self.p_dim], device=self.device), \
                               torch.zeros([1, self.p_dim], device=self.device)

        self.AB_list_new = None

    def lqr_ref(self, step, x, u): # optimal input을 뱉고 linear matrices를 업데이트한다
        if step == 0:
            self.AB_list_md = self.AB_list_new  # 시뮬레이션하기위한 old 모델
            # to update the AB list from the past episode
            if self.AB_list_new is not None:
                self.AB_list_new.reverse()#bakcward pass를 위해 뒤집는다
            self.AB_list_old = self.AB_list_new # ilqr 풀기위한 모델

            self.K_list = self.K_Riccati_ref(x, u, self.AB_list_old)  # 이번 episode(iteration)에서 time-varying policy 가져옴
            self.AB_list_new = []

        l_k, L_k, xd, ud = self.K_list[step] # constant, gain, 선형화 지점 불러옴

        delx = x - xd
        u = ud + (l_k + torch.matmul(L_k, delx.T)).T

        # dfdx, dfdu = self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)
        # A = torch.eye(self.s_dim, device=self.device) + dfdx * self.dt
        # B = dfdu * self.dt


        dfdx, dfdu = self.env.dx_derivs_sym(x[0].detach().numpy(), u[0].detach().numpy())
        dfdx = torch.tensor(dfdx)
        dfdu = torch.tensor(dfdu)
        A = torch.eye(self.s_dim, device=self.device) + dfdx * self.dt
        B = dfdu * self.dt
        A = A.float()
        B = B.float()

        self.AB_list_new.append((A, B, x, u)) # 현재 iteration, step에서 nominal trajectory(다음 epi의 선형화 지점), linear model 업데이트
        return u

    def K_Riccati_ref(self, x, u, AB_list=None):#0~nT-1 까지의 gain 정보가 들어있다
        if AB_list is None:
            dfdx, dfdu = self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)
            A0 = torch.eye(self.s_dim, device=self.device) + dfdx * self.dt
            B0 = dfdu * self.dt
            xd = x
            ud = u
        else:
            A, B, xd, ud = AB_list[0] # time-varying A,B와 nominal trajectory(제일 마지막 step)

        # Riccati equation solving

        qqT, QT = self.cT_derivs(xd) # 1차,2차 미분값을 계산
        qqT = qqT.T

        S = QT
        ss = qqT
        s = torch.zeros([1, 1], device=self.device)
        K_list = []
        for n in range(self.nT):
            if AB_list is None:
                A, B = A0, B0 # nonminal trajectory를 constant(initial point)로 가정
            else:
                A, B, xd, ud = AB_list[n] # time-varying A,B와 nominal trajectory( (nT-n) 번째)


            q0 = self.c_eval(xd, ud) # nominal trajectroy에서 현재 cost값
            qq0, rr0, Q, _, R = self.env.c_derivs_sym(xd[0].detach().numpy(), ud[0].detach().numpy()) # nominal trajectroy에서 cost의 1차, 2차 미분값
            qq0=torch.tensor(qq0).float()
            rr0 = torch.tensor(rr0).float()
            Q = torch.tensor(Q).float()
            R = torch.tensor(R).float()
            # qq0, rr0, Q, _, R = self.c_derivs(xd, ud) # nominal trajectroy에서 cost의 1차, 2차 미분값
            qq0, rr0 = qq0.T, rr0.T

            # print('q', qq0)
            # print('r', rr0)
            # print('Q', Q)
            # print('R', R)

            gg = rr0 + torch.matmul(B.T, ss)#ddp에서 Q_{u}에 해당
            G = torch.chain_matmul(B.T, S, A)#ddp에서 Q_{ux}에 해당, c_{ux}=0이라서 빠져 있다
            H = R + torch.chain_matmul(B.T, S, B)#ddp에서 Q_{uu}에 해당
            H = (H+H.T)/2
            try:
                H_chol = torch.cholesky(H)
            except RuntimeError:
                print('d')
            Hi = torch.cholesky_inverse(H_chol)#ddp에서 inv(Q_{uu})에 해당
            Hi = (Hi + Hi.T) / 2
            ln = -torch.matmul(Hi, gg)# k=-inv(Q_{uu})Q_{u}
            Ln = -torch.matmul(Hi, G)# K=-inv(Q_{uu})Q_{ux}
            K_list.append((ln, Ln, xd, ud))# quadruplet 으로 들어간다

            S = Q + torch.chain_matmul(A.T, S, A) - torch.chain_matmul(G.T, Hi, G)#ddp에서 V_{xx}에 해당
            S = (S + S.T) / 2
            ss = qq0 + torch.matmul(A.T, ss) - torch.chain_matmul(G.T, Hi, gg)#ddp에서 V_{x}에 해당
            s = q0 + s - .5 * torch.chain_matmul(gg.T, Hi, gg)#ddp에서 Delta V(i)에 해당

        K_list.reverse()
        return K_list #0~nT-1 까지의 gain 정보가 들어있다