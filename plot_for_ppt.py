from environment.cstr import CSTR
import numpy as np
import os
import matplotlib.pyplot as plt
from config import get_config, get_env
from utility.plotting import plot_ref

"""traj_data_history: (num_evaluate, NUM_CASE, nT, traj_dim)"""
color_cycle_tab20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
                     '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                     '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
                     '#17becf', '#9edae5']
#%%
config = {'need_derivs': [],
          'need_noise_derivs': [],
          'need_deriv_inverse': []}
env = CSTR(config)
traj_data_history = np.load("_Result\\CSTR_A2C\\traj_data_history.npy")
save_name = os.path.join("_Result\\CSTR_A2C\\ppt_fig")

plot_case = [0] + [200*i-1 for i in range(1, 6)]
case_name = ['Episode ' + str(case+1) for case in range(1000)]

env.plot_info['state_plot_shape'] = (2,2)
env.plot_info['action_plot_shape'] = (2,1)
env.plot_info['state_plot_idx_lst'] = [k+1 for k in range(4)]
env.s_dim = 4

show_plot = True
fs = 24

variable_tag_lst = env.plot_info['variable_tag_lst']
state_plot_idx_lst = env.plot_info['state_plot_idx_lst'] if 'state_plot_idx_lst' in env.plot_info else range(1,
                                                                                                             env.s_dim)
ref_idx_lst = env.plot_info['ref_idx_lst']
nrows_s, ncols_s = env.plot_info['state_plot_shape']
nrows_a, ncols_a = env.plot_info['action_plot_shape']

ref = env.ref_traj()
x_axis = np.linspace(env.t0 + env.dt, env.tT, num=env.nT)

traj_mean = traj_data_history.mean(axis=0)
traj_std = traj_data_history.std(axis=0)

# State variables subplots
fig1, ax1 = plt.subplots(nrows_s, ncols_s, figsize=(ncols_s * 8, nrows_s * 5), sharex='col')
for i, fig_idx in enumerate(ref_idx_lst):
    ax1.flat[fig_idx - 1].hlines(ref[i], env.t0, env.tT, color='r', linestyle='--', label='Set point')

for fig_idx, i in enumerate(state_plot_idx_lst):
    if fig_idx > 1:
        ax1.flat[fig_idx].set_xlabel(variable_tag_lst[0], fontsize=fs)
    ax1.flat[fig_idx].set_ylabel(variable_tag_lst[fig_idx + 1], fontsize=fs)
    ax1.flat[fig_idx].tick_params(labelsize=fs-2)
    if len(plot_case) > 2:
        ax1.flat[fig_idx].set_prop_cycle(color=color_cycle_tab20)
    for case in plot_case:
        ax1.flat[fig_idx].plot(x_axis, traj_mean[case, :, i], label=case_name[case])
        ax1.flat[fig_idx].fill_between(x_axis, traj_mean[case, :, i] + traj_std[case, :, i],
                                       traj_mean[case, :, i] - traj_std[case, :, i], alpha=0.5)
    if i in ref_idx_lst:
        ax1.flat[fig_idx].legend(loc='best', fontsize=14)
    ax1.flat[fig_idx].grid()
    ax1.flat[fig_idx].autoscale(enable=True, axis='x', tight=True)
fig1.tight_layout()
plt.savefig(save_name + '_state_traj.png')
plt.savefig(save_name + '_state_traj.svg')
plt.savefig(save_name + '_state_traj.pdf')
if show_plot:
    plt.show()
plt.close()

# Action variables subplots
x_axis = np.linspace(env.t0, env.tT, num=env.nT)
fig3, ax3 = plt.subplots(nrows_a, ncols_a, figsize=(ncols_a * 8, nrows_a * 5), sharex='col')
for i in range(env.a_dim):
    axis = ax3.flat[i] if env.a_dim > 1 else ax3
    if i >= ncols_a:
        axis.set_xlabel(variable_tag_lst[0], fontsize=fs)
    axis.set_ylabel(variable_tag_lst[len(state_plot_idx_lst) + i + 1], fontsize=fs)
    axis.tick_params(labelsize=fs-2)
    if len(plot_case) > 2:
        axis.set_prop_cycle(color=color_cycle_tab20)
    for case in plot_case:
        axis.plot(x_axis, traj_mean[case, :, env.s_dim + i], label=case_name[case])
        axis.fill_between(x_axis, traj_mean[case, :, env.s_dim + i] + traj_std[case, :, env.s_dim + i],
                          traj_mean[case, :, env.s_dim + i] - traj_std[case, :, env.s_dim + i], alpha=0.5)
    if i < 1:
        axis.legend(fontsize=14)
    axis.grid()
    axis.autoscale(enable=True, axis='x', tight=True)
fig3.tight_layout()
plt.savefig(save_name + '_action_traj.png')
plt.savefig(save_name + '_action_traj.svg')
plt.savefig(save_name + '_action_traj.pdf')
if show_plot:
    plt.show()
plt.close()


#%%
env_agent = "CSTR_A2C"
learning_stat_history = np.load("_Result\\CSTR_A2C\\learning_stat_history.npy")
learning_stat_lst = ['Cost', 'Convergence criteria']
save_name = "_Result\\CSTR_A2C\\ppt_fig"

nrows, ncols = 2,1
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 5), sharex='col')

learning_stat_dim = 2
for i in range(learning_stat_dim):
    ax.flat[i].plot(learning_stat_history[:, i])
    if i >= ncols:
        ax.flat[i].set_xlabel('Episode', size=fs)
    ax.flat[i].set_ylabel(learning_stat_lst[i], size=fs)
    ax.flat[i].tick_params(labelsize=fs-2)
    ax.flat[i].grid()
fig.tight_layout()
plt.savefig(save_name + f'_{env_agent}_stats_plot.png')
plt.show()
plt.close()


#%%
from environment.crystallization import CRYSTAL
env_alg = 'CRYSTAL_DDPG'
env_name = 'CRYSTAL'
traj_data_history = np.load(os.path.join("_Result", env_alg, "traj_data_history.npy"))
learning_stat_history = np.load(os.path.join("_Result", env_alg, "learning_stat_history.npy"))

# crystal_growth = np.expand_dims(traj_data_history[:,:,:,4] / traj_data_history[:,:,:,3], axis=3)
# traj_data_history = np.concatenate([traj_data_history, crystal_growth], axis=-1)
# ref_idx = -1

plot_case = [0] + [100*i-1 for i in range(1, 6) if 100*i-1 < np.size(traj_data_history, 1)]
case_name = ['Episode ' + str(case+1) for case in range(1000)]

save_name = os.path.join("_Result", env_alg, "ppt_fig_ref_cc")

config = {'env': env_name,
          'need_derivs': [],
          'need_noise_derivs': [],
          'need_deriv_inverse': []}
env = get_env(config)

# variable_tag_lst = env.plot_info['variable_tag_lst']
# variable_tag_lst.append(r'Crystal growth')
# x_axis = np.linspace(env.t0 + env.dt, env.tT, num=env.nT)
#
# traj_mean = traj_data_history.mean(axis=0)
# traj_std = traj_data_history.std(axis=0)

fig, ax = plt.subplots(1, 2, figsize=(2 * 10, 1 * 5))
fs = 24

learning_stat_lst = ['Cost', 'Convergence criteria']
for i in range(2):
    ax_conv = ax.flat[i]
    ax_conv.plot(learning_stat_history[:, i])
    ax_conv.set_xlabel('Episode', size=fs)
    ax_conv.set_ylabel(learning_stat_lst[i], size=fs)
    ax_conv.tick_params(labelsize=fs-2)
    ax_conv.grid()

# ax_ref = ax.flat[1]
# ax_ref.set_xlabel(variable_tag_lst[0], fontsize=fs)
# ax_ref.set_ylabel(variable_tag_lst[ref_idx], fontsize=fs)
# ax_ref.tick_params(labelsize=fs-2)
# for case in plot_case:
#     ax_ref.plot(x_axis, traj_mean[case, :, ref_idx], label=case_name[case])
#     ax_ref.fill_between(x_axis, traj_mean[case, :, ref_idx] + traj_std[case, :, ref_idx],
#                      traj_mean[case, :, ref_idx] - traj_std[case, :, ref_idx], alpha=0.5)
# ax_ref.legend(loc='upper right', bbox_to_anchor=(1.3, 1.), fontsize=14)
# # ax_ref.legend(loc='best', fontsize=14)
# ax_ref.grid()
# ax_ref.set_xlim((env.t0, env.tT))

fig.tight_layout()

plt.savefig(save_name + f'_{env_alg}_stats_only_plot.png')
plt.show()

#%%
import glob

env_name = 'CRYSTAL'

available_file_path = glob.glob(f'./_Result/test_{env_name}_*')
available_alg_names = []
for dir_path in available_file_path:
    dir_name = os.path.basename(dir_path)
    if os.path.isfile(os.path.join(dir_path, 'test_traj_data_history.npy')):
        alg_name = dir_name[6 + len(env_name):]
        available_alg_names.append(alg_name)

# highlight_alg_names = available_alg_names
highlight_color = color_cycle_tab20

highlight_alg_names = ["GDHP", "SDDP", "SAC"]
# highlight_color = []
# for alg_name in highlight_alg_names:
#     alg_idx = available_alg_names.index(alg_name)
#     highlight_color.append(color_cycle_tab20[alg_idx])

save_path = os.path.join(os.getcwd(), '_Result', f'test_all_{env_name}')
show_plot = True

config = {'env': env_name,
          'save_path': save_path,
          'need_derivs': [],
          'need_noise_derivs': [],
          'need_deriv_inverse': []}
env = get_env(config)

traj_data_history = []
for alg_name in highlight_alg_names:
    test_path = os.path.join(f'./_Result/test_{env_name}_{alg_name}')
    traj = np.load(os.path.join(test_path, 'test_traj_data_history.npy'))
    traj_data_history.append(traj)
traj_data_history = np.concatenate(traj_data_history, axis=1)

save_name = os.path.join(config['save_path'], 'ppt_fig3_highlight')
plot_case = [i for i in range(len(highlight_alg_names))]
variable_tag_lst = env.plot_info['variable_tag_lst']

# ref_idx = env.plot_info['ref_idx_lst'][0]
# ref = env.ref_traj()

# ref_idx = 7
# ref = [[]]

traj_data_history, variable_tag_lst, ref_idx_lst, ref = env.max_variable_info(traj_data_history)
ref_idx = ref_idx_lst[0]

x_axis = np.linspace(env.t0 + env.dt, env.tT, num=env.nT)

traj_mean = traj_data_history.mean(axis=0)
traj_std = traj_data_history.std(axis=0)

fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5), layout='constrained')
fs = 22

ax1.hlines(ref[0], env.t0, env.tT, color='r', linestyle='--', label='Set point')
ax1.set_xlabel(variable_tag_lst[0], fontsize=fs)
ax1.set_ylabel(variable_tag_lst[ref_idx], fontsize=fs)
ax1.tick_params(labelsize=fs - 2)
ax1.set_prop_cycle(color=highlight_color)
for case in plot_case:
    ax1.plot(x_axis, traj_mean[case, :, ref_idx], label=highlight_alg_names[case])
    ax1.fill_between(x_axis, traj_mean[case, :, ref_idx] + traj_std[case, :, ref_idx],
                     traj_mean[case, :, ref_idx] - traj_std[case, :, ref_idx], alpha=0.5)
ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 0.99))
ax1.grid()
ax1.set_xlim((env.t0, env.tT))
# ax1.set_ylim((1.65, 2.65))
# fig1.tight_layout()
plt.savefig(save_name + f'_{str(ref_idx)}.png')
plt.savefig(save_name + f'_{str(ref_idx)}.svg')
plt.savefig(save_name + f'_{str(ref_idx)}.pdf')
plt.show()
plt.close()


#%%
from utility.plotting import plot_traj_data
env_name = 'DISTILLATION'
alg_name = 'DQN'
env_alg = env_name + '_' + alg_name
config = {'env': env_name,
          'save_path': [],
          'need_derivs': [],
          'need_noise_derivs': [],
          'need_deriv_inverse': []}
env = get_env(config)

traj_data_history = np.load(os.path.join("_Result", env_alg, "traj_data_history.npy"))
epi_intv = 10
plot_case = [0] + [epi_intv*i-1 for i in range(1, 6) if epi_intv*i-1 < np.size(traj_data_history, 1)]
case_name = ['Episode ' + str(case+1) for case in range(1000)]
save_name = os.path.join("_Result", env_alg, "ppt_fig2")

plot_ref(env, traj_data_history, plot_case, case_name, save_name, True, opt='Production')
# plot_traj_data(env, traj_data_history, plot_case, case_name, save_name, True)

#%%
from performance_summary import get_summary_history
from utility.plotting import plot_radar

available_algs, available_envs, summary_path, summary_feature, history, history_with_name = get_summary_history()

scaling_factor = {"min": [min([h[f][0] for h in history]) for f in range(len(summary_feature))],
                  "max": [max([h[f][0] for h in history]) for f in range(len(summary_feature))]}
scaling_factor["scale"] = [max(scaling_factor["max"][f] - scaling_factor["min"][f], 0.01) for f in
                           range(len(summary_feature))]

radar_algs = ['GDHP', 'SDDP', 'DDPG', 'SAC']
radar_data = []
for alg_name in radar_algs:
    i = available_algs.index(alg_name)
    alg_data = [(history[i][f][0] - scaling_factor["min"][f]) / scaling_factor["scale"][f]
                for f in range(len(summary_feature))]
    radar_data.append(alg_data)
    alg_filename = os.path.join(summary_path, f'{alg_name}')

filename = os.path.join(summary_path, 'radar_top6')
plot_radar(radar_data, radar_algs, summary_feature, filename)
