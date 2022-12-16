# 수정된 모든 모듈을 reload
%load_ext autoreload
%autoreload 2

import utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
my_CSTR = os.getcwd()
sys.path.append(my_CSTR)
from env import CstrEnv
from gps import GPS
import time
import copy
import gc
from random import *
import psutil

MAX_ITERATION = 1 # C,S step 몇번 진행하는지
MAX_COND = 1 # initial condition의 갯수
MAX_EPISODE = 1 # 같은 initial condtion에서 몇번 뽑을 것인지(stochastic environment의 경우)

device = 'cpu'

env = CstrEnv(device)
gps_controller = GPS(env, device)
# ilqr_controller = Ilqr(env, device)

s_dim = env.s_dim
a_dim = env.a_dim
o_dim = env.o_dim

plt_num = 0

def get_tensor_info(tensor):
  info = []
  for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
    info.append(f'{name}({getattr(tensor, name, None)})')
  info.append(f'tensor({str(tensor)})')
  return ' '.join(info)

# 학습 과정
start_time = time.time()
for ii in range(0,200*6*10*10): # it*env.t/n_mini*n_cond
    kl_div = torch.tensor([[0.]])
    u_all=torch.zeros([1,a_dim])
    ug_all=torch.zeros([1,a_dim])
    for sample in range(30):
        cond=randint(21,30) # 학습시키고 싶은 cases들(각각 다른 초기조건들)를 지정
        step=randint(1,180)
        path=my_CSTR+'/data/ilqr2/initial_'+str(cond) # 각 초기조건 마다의 결과가 저장된 디렉토리
        traj=np.loadtxt(path+'/cstr_trajectory15.txt')
        traj=traj[1:]
        Klist=torch.load(path+'/klist.pt')

        l_k, L_k, xd, ud = Klist[step] # constant, gain, 선형화 지점 불러옴
        x=torch.tensor(traj[step,:7]).float()

        delx = x - xd
        u = ud + (l_k + torch.matmul(L_k, delx.T)).T


        u_g = gps_controller.p_net(x)
        u_g = torch.reshape(u_g,(1,a_dim))
        u_all=torch.cat((u_all,u))
        ug_all=torch.cat((ug_all,u_g))
        kl_div=kl_div+torch.chain_matmul((u - u_g),  (u - u_g).T)
    ug_all=ug_all[1:].detach().numpy()
    u_all=u_all[1:].detach().numpy()

    if ii%10000 == 0:
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        fig = plt.figure(figsize=[20, 12])
        fig.subplots_adjust(hspace=.4, wspace=.5)
        label = [r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$']
        for j in range(len(label)):
            ax = fig.add_subplot(2, 2, j+1)
            ax.save(u_all[:, j])
            plt.ylabel(label[j], size=30)
            if j==0:
                ax.set_ylim([-0.1, 1.75])
            else:
                ax.set_ylim([-0.6, 0.2])
        for j in range(len(label)):
            ax = fig.add_subplot(2, 2, j+3)
            ax.save(ug_all[:, j])
            plt.ylabel(label[j], size=30)
            if j==0:
                ax.set_ylim([-0.1, 1.75])
            else:
                ax.set_ylim([-0.6, 0.2])
        plt.savefig(my_CSTR + '/data/ilqr2/21to30_ilqr' + str(ii) + '.png')
        plt.clf()
        plt.close()
        gc.collect()
        print(ii,'th kl_div : ', kl_div)
    gps_controller.s_step(kl_div)
elapsed_time = time.time() - start_time
print(elapsed_time)

# # 학습된 모델 저장
# torch.save(gps_controller.p_net.state_dict(),'gps_p_net_weight.pt')

# 학습된 모델 불러오기
gps_controller_learned = GPS(env, device)
gps_controller_learned.p_net.load_state_dict(torch.load('gps_p_net_weight.pt'))
gps_controller_learned.p_net.eval()


# 검증 과정 1(각 trajectory에서 input 비교- ons-step prediction 능력)
start_time = time.time()
for cond in range(21,30):
    kl_div = torch.tensor([[0.]])
    u_all=torch.zeros([1,a_dim])
    ug_all=torch.zeros([1,a_dim])
#     cond=randint(21,30)
#     cond=23
    print(cond,'th initial state')
    for sample in range(env.nT):

        step=sample
        path=my_CSTR+'/data/ilqr2/initial_'+str(cond)
        traj=np.loadtxt(path+'/cstr_trajectory15.txt')
        traj=traj[1:]
        Klist=torch.load(path+'/klist.pt')

        l_k, L_k, xd, ud = Klist[step] # constant, gain, 선형화 지점 불러옴
        x=torch.tensor(traj[step,:7]).float()

        delx = x - xd
        u = ud + (l_k + torch.matmul(L_k, delx.T)).T

    #     u_g = gps_controller.p_net(x)
        u_g = gps_controller_learned.p_net(x)
        u_g = torch.reshape(u_g,(1,a_dim))
        u_all=torch.cat((u_all,u))
        ug_all=torch.cat((ug_all,u_g))
        kl_div=kl_div+torch.chain_matmul((u - u_g),  (u - u_g).T)
    ug_all=ug_all[1:].detach().numpy()
    u_all=u_all[1:].detach().numpy()


    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    fig = plt.figure(figsize=[20, 12])
    fig.subplots_adjust(hspace=.4, wspace=.5)
    label = [r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$']
    for j in range(len(label)):
        ax = fig.add_subplot(2, 2, j+1)
        ax.save(u_all[:, j])
        plt.ylabel(label[j], size=30)
        if j==0:
            ax.set_ylim([-0.1, 2])
        else:
            ax.set_ylim([-1, 0.2])
    for j in range(len(label)):
        ax = fig.add_subplot(2, 2, j+3)
        ax.save(ug_all[:, j])
        plt.ylabel(label[j], size=30)
        if j==0:
            ax.set_ylim([-0.1, 2])
        else:
            ax.set_ylim([-1, 0.2])
    # plt.savefig(my_CSTR + '/data/ilqr2/21to30_ilqr' + str(ii) + '.png')
    # plt.clf()
    # plt.close()
    # gc.collect()
    print(cond,'th kl_div : ', kl_div)

elapsed_time = time.time() - start_time
print(elapsed_time)

# 검증 과정 2(각 initial condition과 플랜트에서 state 비교 full-step prediction 능력)
# ilqr input 사용
start_time = time.time()
cond = 21
print(cond, 'th initial point')
path = my_CSTR + '/data/ilqr2/initial_' + str(cond)

Klist = torch.load(path + '/klist.pt')
x0, y0, u0, data_type = env.reset()
x0 = torch.load(path + '/x0.pt')

x, y, u, data_type = x0, y0, u0, data_type
u_idx = None
trajectory = torch.zeros([1, s_dim + o_dim + a_dim + 2], device=device)  # s + o + a + r + ref

for i in range(env.nT):
    # 1. ilqr
    l_k, L_k, xd, ud = Klist[i]  # constant, gain, 선형화 지점 불러옴
    delx = x - xd
    u = ud + (l_k + torch.matmul(L_k, delx.T)).T
    # 2. gps
    #     u=gps_controller_learned.p_net(x)

    x2, y2, u, r, is_term, derivs = env.step(x, u)

    xu = torch.cat((x.squeeze(0), u.squeeze(0)))
    xuy = torch.cat((xu, torch.reshape(y2, [-1, ])))
    ref = utils.scale(env.ref_traj(), env.ymin, env.ymax).squeeze(0)
    xuyr = torch.cat((xuy, r.squeeze(0)))
    xuyrr = torch.cat((xuyr, ref))
    xuyrr = torch.reshape(xuyrr, [1, -1])
    trajectory = torch.cat((trajectory, xuyrr))

    x, y = x2, y2

trajectory_n = trajectory.detach().cpu().numpy()

#     np.savetxt(my_CSTR + '/data/ilqr2/initial_'+str(it)+'/cstr_trajectory' + str(epi) + '.txt', trajectory_n, newline='\n')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
fig = plt.figure(figsize=[20, 12])
fig.subplots_adjust(hspace=.4, wspace=.5)
label = [r'$C_{A}$', r'$C_{B}$', r'$T$', r'$T_{Q}$', r'$\frac{v}{V_{R}}$', r'$Q$',
         r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$', r'$C_{B}$', r'$cost$']
for j in range(len(label)):
    if label[j] in (r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$'):
        ax = fig.add_subplot(2, 6, j + 5)
    else:
        ax = fig.add_subplot(2, 6, j + 1)
    ax.save(trajectory_n[1:, 0], trajectory_n[1:, j + 1])
    if j in (1, 8):
        ax.save(trajectory_n[1:, 0], trajectory_n[
                                     1:, -1], ':g')
    plt.ylabel(label[j], size=8)
#     plt.savefig(my_CSTR + '/data/ilqr2/initial_'+str(it)+'/episode' + str(epi) + '.png')
#     plt.clf()
#     plt.close()
#     gc.collect()


elapsed_time = time.time() - start_time
print(elapsed_time)