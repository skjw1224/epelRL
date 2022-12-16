import utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
my_CSTR = os.getcwd()
sys.path.append(my_CSTR)
from env import CstrEnv
from pi import PI
from pi import InitialControl

MAX_EPISODE = 1

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
device = 'cpu'

env = CstrEnv(device)
initial_controller = InitialControl()
PI_controller = PI(env, initial_controller, device)

s_dim = env.s_dim
a_dim = env.a_dim
o_dim = env.o_dim

xmin = env.xmin
xmax = env.xmax
umin = env.umin
umax = env.umax

plt_num = 0

for epi in range(MAX_EPISODE):
    x, y, u, data_type = env.reset()
    u_idx = None
    trajectory = torch.zeros([1, s_dim + o_dim + a_dim + 2], device=device)  # s + o + a + r + ref

    for i in range(env.nT):
        print("time: ", i)
        u = PI_controller.ctrl(epi, i, x, u)
        x2, y2, u, r, is_term, derivs = env.step(x, u)

        # Adding Brownian motion
        x_descaled = utils.descale(x, xmin, xmax)
        Browian_motion = PI_controller.B_torch(x_descaled)
        w = torch.normal(0, 1, size=(1, env.a_dim))
        x2_descaled = utils.descale(x2, xmin, xmax)
        x2_descaled += w @ Browian_motion.T * torch.sqrt(torch.tensor(env.dt))
        x2 = utils.scale(x2_descaled, xmin, xmax)

        u_descaled = utils.descale(u, umin, umax)
        # xu = torch.cat((x.squeeze(0), u.squeeze(0)))
        xu_descaled = torch.cat((x_descaled.squeeze(0), u_descaled.squeeze(0)))
        # xuy = torch.cat((xu, torch.reshape(y2, [-1, ])))
        xuy = torch.cat((xu_descaled, torch.reshape(y2, [-1, ])))
        ref = utils.scale(env.ref_traj(), env.ymin, env.ymax).squeeze(0)
        xuyr = torch.cat((xuy, r.squeeze(0)))
        xuyrr = torch.cat((xuyr, ref))
        xuyrr = torch.reshape(xuyrr, [1, -1])
        trajectory = torch.cat((trajectory, xuyrr))

        x, y = x2, y2

    trajectory_n = trajectory.detach().cpu().numpy()

    np.savetxt('trajectory.csv', trajectory_n, delimiter=",", newline='\n')

    # if epi%5 == 0:
    #     np.savetxt(my_CSTR + '/data/cstr_trajectory' + str(epi) + '.txt', trajectory_n, newline='\n')
    #     plt.rc('xtick', labelsize=8)
    #     plt.rc('ytick', labelsize=8)
    #     fig = plt.figure(figsize=[20, 12])
    #     fig.subplots_adjust(hspace=.4, wspace=.5)
    #     label = [r'$C_{A}$', r'$C_{B}$', r'$T$', r'$T_{Q}$', r'$\frac{v}{V_{R}}$', r'$Q$',
    #              r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$', r'$C_{B}$', r'$cost$']
    #     for j in range(len(label)):
    #         if label[j] in (r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$'):
    #             ax = fig.add_subplot(2, 6, j+5)
    #         else:
    #             ax = fig.add_subplot(2, 6, j+1)
    #         ax.plot(trajectory_n[1:, 0], trajectory_n[1:, j+1])
    #         if j in (1, 8):
    #             ax.plot(trajectory_n[1:, 0], trajectory_n[1:, -1], ':g')
    #         plt.ylabel(label[j], size=8)
    #     plt.savefig(my_CSTR + '/data/episode' + str(epi) + '.png')
    #     # if epi%5 == 0: plt.show()
    #     plt_num += 1
    # print(epi)
