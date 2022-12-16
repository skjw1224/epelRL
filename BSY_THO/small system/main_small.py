import utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
my_CSTR = os.getcwd()
sys.path.append(my_CSTR)
from ENV_small import Env
from lqr_small import lqr
from AAC_tho import Actor_Critic
from lstd_small import LSTD_small

MAX_EPISODE = 100

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
device = 'cpu'

env = Env(device)
lqr_controller = lqr(env, device)
aac_controller = Actor_Critic(env, device)
lstd_controller = LSTD_small(env, device)

s_dim = env.s_dim
a_dim = env.a_dim
o_dim = env.o_dim

plt_num = 0

for epi in range(MAX_EPISODE):
    x, y, u, data_type = env.reset()
    u_idx = None
    # trajectory = torch.zeros([1, s_dim + o_dim + a_dim + 2], device=device)  # A2C
    time_track = torch.tensor([0.], device=device)
    cost = 0 ##
    trajectory = torch.zeros([1, s_dim + a_dim + 1], device=device)  # LQR
    # K0 = lqr_controller.LQR_K_list[0]   # LQR
    # u = K0 * x                          # LQR
    print("episode: ", epi)

    for i in range(env.nT):

        """A2C"""
        # x2, y2, u, r, is_term, derivs = env.step(x, u, i)
        # cost += r
        # u = aac_controller.ctrl(epi, i, x, u, r, x2, is_term, derivs)
        # print("time: ", i, "state: ", x)
        #
        # x_descaled = utils.descale(x, env.xmin, env.xmax)
        # u_descaled = utils.descale(u, env.umin, env.umax)
        # y2_descaled = utils.descale(y2, env.ymin, env.ymax)
        # xu = torch.cat((x_descaled.squeeze(0), u_descaled.squeeze(0)))
        # xuy = torch.cat((xu, torch.reshape(y2_descaled, [-1, ])))
        # ref = env.ref_traj()
        # xuyr = torch.cat((xuy, r.squeeze(0)))
        # xuyrr = torch.cat((xuyr, ref))
        # xuyrr = torch.reshape(xuyrr, [1, -1])
        # trajectory = torch.cat((trajectory, xuyrr))
        # time_track = torch.cat((time_track, x[0][0].unsqueeze(0)))
        # x, y = x2, y2

        # """LQR"""
        # x2, y2, u, r, is_term, derivs = env.step(x, u, i)
        # cost += r
        # xu = torch.cat((x, u))
        # xur = torch.cat((xu, r))
        # xur = torch.reshape(xur, [1, 3])
        # if is_term:
        #     u = 0
        # else:
        #     u = lqr_controller.ctrl(i, x2, is_term)
        # trajectory = torch.cat([trajectory, xur], dim=0)
        # x, y = x2, y2

        """LSTD"""
        x2, y2, u, r, is_term, derivs = env.step(x, u, i)
        cost += r
        u = lstd_controller.ctrl(epi, i, x, u, r, x2, is_term, derivs)
        x_saving = x.squeeze(0)
        r_saving = r.squeeze(0)
        xu = torch.cat((x_saving, u))
        xur = torch.cat((xu, r_saving))
        xur = torch.reshape(xur, [1, 3])
        trajectory = torch.cat((trajectory, xur))
        x, y = x2, y2

    print(cost) ##
    # trajectory_n = trajectory.detach().cpu().numpy()  # A2C
    # time_track_n = time_track.detach().cpu().numpy()  # A2C
    trajectory = trajectory.detach().cpu().numpy()
    np.savetxt('trajectory' + str(epi) + '.csv', trajectory, delimiter=",", newline='\n')   # A2C

    # np.savetxt('trajectory.csv', trajectory, delimiter=",", newline='\n')  # LQR
    # print(epi)