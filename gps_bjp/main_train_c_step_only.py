%load_ext autoreload
%autoreload 2
import utils
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
my_CSTR = os.getcwd()
sys.path.append(my_CSTR)
from env import CstrEnv
from ilqr2 import Ilqr
from pid import PID
from dqn import DQN
from gdhp import GDHP
from ddpg import DDPG
from a2c import Actor_Critic
from trpo_modif import TRPO
from PoWER import PoWER

import time
import gc

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set(style="white", font="Arial")
colors = sns.color_palette("Paired", n_colors=12).as_hex()
from gmm_edit import GaussianMixture

MAX_EPISODE = 16

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
device = 'cpu'

env = CstrEnv(device)
ilqr_controller = Ilqr(env, device)
pid_controller = PID(env, device)
dqn_controller = DQN(env, device)
gdhp_controller = GDHP(env, device)
ddpg_controller = DDPG(env, device)
trpo_controller = TRPO(env, device)
a2c_controller = Actor_Critic(env, device)
PoWER_controller = PoWER(env, device)

s_dim = env.s_dim
a_dim = env.a_dim
o_dim = env.o_dim

plt_num = 0

# 초기조건마다 ilqr controller 생성(gps controller가 학습할 것들)
start_time = time.time()
for it in range(70, 71):  #
    x0, y0, u0, data_type = env.reset()
    # it 마다 초기조건을 다르게 지정하여 ilqr 생성
    #     x0=x0*torch.normal(mean=x0.detach()*0+1,std=0.1*torch.tensor([[0., 1., 1., 1., 1., 1., 1.]], dtype=torch.float))
    print('it:', str(it), ', x0: ', x0)
    ilqr_controller = Ilqr(env, device)
    path = my_CSTR + '/data/ilqr2/initial_' + str(it)
    if not (os.path.isdir(path)):
        os.mkdir(path)
    for epi in range(1):
        x, y, u, data_type = x0, y0, u0, data_type
        u_idx = None
        trajectory = torch.zeros([1, s_dim + o_dim + a_dim + 2], device=device)  # s + o + a + r + ref
        sas = torch.zeros([1, 2 * s_dim + a_dim], device=device)
        for i in range(env.nT):
            u = ilqr_controller.lqr_ref(i, x, u)

            x2, y2, u, r, is_term, derivs = env.step(x, u)

            xu = torch.cat((x.squeeze(0), u.squeeze(0)))
            xuy = torch.cat((xu, torch.reshape(y2, [-1, ])))
            ref = utils.scale(env.ref_traj(), env.ymin, env.ymax).squeeze(0)
            xuyr = torch.cat((xuy, r.squeeze(0)))
            xuyrr = torch.cat((xuyr, ref))
            xuyrr = torch.reshape(xuyrr, [1, -1])
            trajectory = torch.cat((trajectory, xuyrr))

            xux = torch.cat((xu, x2.squeeze(0)))
            xux = torch.reshape(xux, [1, -1])
            sas = torch.cat((sas, xux))
            x, y = x2, y2

        trajectory_n = trajectory.detach().cpu().numpy()

        if epi % 5 == 0:
            np.savetxt(my_CSTR + '/data/ilqr2/initial_' + str(it) + '/cstr_trajectory' + str(epi) + '.txt',
                       trajectory_n, newline='\n')
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
                ax.plot(trajectory_n[1:, 0], trajectory_n[1:, j + 1])
                if j in (1, 8):
                    ax.plot(trajectory_n[1:, 0], trajectory_n[
                                                 1:, -1], ':g')
                plt.ylabel(label[j], size=8)
            plt.savefig(my_CSTR + '/data/ilqr2/initial_' + str(it) + '/episode' + str(epi) + '.png')
            plt.clf()
            plt.close()
            gc.collect()
            # if epi%5 == 0: plt.show()
            plt_num += 1
        print('epi: ', epi)
    klist = ilqr_controller.K_list
    torch.save(klist, my_CSTR + '/data/ilqr2/initial_' + str(it) + '/klist.pt')
    torch.save(x0, my_CSTR + '/data/ilqr2/initial_' + str(it) + '/x0.pt')
    print('it: ', it)
elapsed_time = time.time() - start_time
print(elapsed_time)

trajectory_n =np.loadtxt(my_CSTR + '/data/ilqr2/initial_'+str(0)+'/cstr_trajectory' + str(0) + '.txt')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

fig = plt.figure(figsize=[20, 12])
fig.subplots_adjust(hspace=.4, wspace=.5)
label = [r'$C_{A}$', r'$C_{B}$', r'$T$', r'$T_{Q}$', r'$\frac{v}{V_{R}}$', r'$Q$',
         r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$', r'$C_{B}$', r'$cost$']
for j in range(len(label)):
    if label[j] in (r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$'):
        ax = fig.add_subplot(2, 6, j+5)
    else:
        ax = fig.add_subplot(2, 6, j+1)
    ax.plot(trajectory_n[1:, 0], trajectory_n[1:, j+1])
    if j in (1, 8):
        ax.plot(trajectory_n[1:, 0], trajectory_n[
            1:, -1], ':g')
    plt.ylabel(label[j], size=15)


plt.savefig(my_CSTR + '/data/ilqr2/initial_'+str(0)+'/episode' + str(0) + '.png')