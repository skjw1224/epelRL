import utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
my_CSTR = os.getcwd()
sys.path.append(my_CSTR)
from env import CstrEnv
from pid import PID

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
device = 'cpu'

env = CstrEnv(device)

s_dim = env.s_dim
a_dim = env.a_dim
o_dim = env.o_dim

plt_num = 0
K1_list = np.linspace(61,70,10)
K2_list = np.linspace(26,35,10)
epi = 0
# the best: (67,33)
for K1 in K1_list:
    for K2 in K2_list:
        K = torch.tensor([[K1], [K2]], requires_grad=False, dtype=torch.float)
        pid_controller = PID(env, device, K_val=K)
        t, x, y, u, data_type = env.reset()
        u_idx = None
        trajectory = torch.zeros([1, s_dim + o_dim + a_dim + 2], device=device)  # s + o + a + r + ref
        time_track = torch.tensor([0.], device=device)

        for i in range(env.nT):
            # u = ilqr_controller.lqr_ref(i, t, x, u)
            u = pid_controller.pid_ctrl(i, t, x)
            print(u)
            t2, x2, y2, u, r, is_term, derivs = env.step(t, x, u)
            # u_idx, u = dqn_controller.dqn_ctrl(epi, i, t, x, u_idx, r, t2, x2)
            # u = gdhp_controller.gdhp_ctrl(epi, i, t, x, u, r, t2, x2, is_term, derivs)
            # u = ddpg_controller.ctrl(epi, i, t, x, u, r, t2, x2, is_term, derivs)
            # u = qrdqn_controller.ctrl(epi, i, t, x, u, r, t2, x2, is_term, derivs)

            xu = torch.cat((x.squeeze(0), u.squeeze(0)))
            xuy = torch.cat((xu, torch.reshape(y2, [-1, ])))
            ref = utils.scale(env.ref_traj(t), env.ymin, env.ymax).squeeze(0)
            xuyr = torch.cat((xuy, r.squeeze(0)))
            xuyrr = torch.cat((xuyr, ref))
            xuyrr = torch.reshape(xuyrr, [1, -1])
            trajectory = torch.cat((trajectory, xuyrr))
            time_track = torch.cat((time_track, t2.squeeze(0)))

            t, x, y = t2, x2, y2

        trajectory_n = trajectory.detach().cpu().numpy()
        time_track_n = time_track.detach().cpu().numpy()

        np.savetxt(my_CSTR + '/data/cstr_trajectory/epi' + str(epi) + '.txt', trajectory_n, newline='\n')
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
            ax.save(time_track_n[1:], trajectory_n[1:, j])
            if j in (1, 8):
                ax.save(time_track_n[1:], trajectory_n[1:, -1], ':g')
                plt.title('K value: ' + str(K.numpy()))
            plt.ylabel(label[j], size=8)
        plt.savefig(my_CSTR + '/data/episode' + str(epi) + '.png')
        # if epi%5 == 0: plt.show()
        plt.close(fig)
        plt_num += 1
        print(epi)
        epi += 1
