import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

import time
my_CSTR = os.getcwd()
sys.path.append(my_CSTR)
# from env import CstrEnv
from env_casadi import CstrEnv
from replay_buffer import ReplayBuffer
from ilqr import Ilqr
from sddp import SDDP
from pid import PID
from dqn import DQN
from gdhp import GDHP
from ddpg import DDPG
from a2c import Actor_Critic
from trpo_modif import TRPO
from PoWER import PoWER



MAX_EPISODE = 21
BUFFER_SIZE = 600
MINIBATCH_SIZE = 32

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
device = 'cpu'

env = CstrEnv(device)
replay_buffer = ReplayBuffer(env, device, buffer_size=BUFFER_SIZE, batch_size=MINIBATCH_SIZE)
ilqr_controller = Ilqr(env, device)
sddp_controller = SDDP(env, device)
pid_controller = PID(env, device)
dqn_controller = DQN(env, device)
gdhp_controller = GDHP(env, device)
ddpg_controller = DDPG(env, device, replay_buffer)
trpo_controller = TRPO(env, device)
a2c_controller = Actor_Critic(env, device)
PoWER_controller = PoWER(env, device)

s_dim = env.s_dim
a_dim = env.a_dim
o_dim = env.o_dim

plt_num = 0

controller = ilqr_controller
controller = ddpg_controller
controller = sddp_controller
controller = pid_controller
controller = dqn_controller
controller = gdhp_controller

for epi in range(MAX_EPISODE):
    t, x, y, u, data_type = env.reset()
    u_idx = None
    trajectory = np.zeros([1, s_dim + o_dim + a_dim + 2])  # s + o + a + r + ref

    for i in range(env.nT):
        u = controller.ctrl(epi, i, x, u)
        u_idx, u = dqn_controller.ctrl(epi, i, x, u_idx, r, x2, is_term)

        # u = trpo_controller.ctrl(epi, i, x, u, r, x2, is_term, derivs)
        # u = a2c_controller.ctrl(epi, i, x, u)
        # u = PoWER_controller.ctrl(epi, i, x, u)

        t2, x2, y2, u, r, is_term, derivs = env.step(t, x, u)
        # print("ode time:", time.time() - start_time)
        # a2c_controller.add_experience(x, u, r, x2, is_term)
        # PoWER_controller.add_experience(x, u, r, x2, is_term)
        # gdhp_controller.add_experience(x, u, r, x2, is_term, derivs)

        controller.add_experience(x, u, r, x2, is_term)

        x_record = np.reshape(x, [1, -1])
        u_record = np.reshape(u, [1, -1])
        y_record = np.reshape(y2, [1, -1])
        r_record = np.reshape(r, [1, -1])
        ref_record = np.reshape(env.scale(env.ref_traj(), env.ymin, env.ymax), [1, -1])
        step_data = np.concatenate([x_record, u_record, y_record, r_record, ref_record], axis=1)
        trajectory = np.concatenate([trajectory, step_data], axis=0)

        t, x = t2, x2

    if epi%5 == 0:
        np.savetxt(my_CSTR + '/data/cstr_trajectory' + str(epi) + '.txt', trajectory, newline='\n')
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        fig = plt.figure(figsize=[20, 12])
        fig.subplots_adjust(hspace=.4, wspace=.5)
        label = [r'$C_{A}$', r'$C_{B}$', r'$T$', r'$T_{Q}$', r'$\frac{v}{V_{R}}$', r'$Q$',
                 r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$', r'$C_{B}$', r'$cost$']
        for j in range(len(label)):
            if label[j] in (r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$'):
                ax = fig.add_subplot(2, 6, j+5)
            else:
                ax = fig.add_subplot(2, 6, j+1)
            ax.plot(trajectory[1:, j+1])
            if j in (1, 8):
                ax.plot(trajectory[1:, -1], ':g')
            plt.ylabel(label[j], size=8)
        plt.savefig(my_CSTR + '/data/episode' + str(epi) + '.png')
        plt.show()
        plt_num += 1
