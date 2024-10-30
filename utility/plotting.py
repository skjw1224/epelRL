import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_radar(data, label, feature, filename, title=''):
    """ data = (N_label, N_feature) """
    cat = feature
    cat = [*cat, cat[0]]
    data = [[*d, d[0]] for d in data]

    cat_linebreak = [c.replace(' ', '\n') for c in cat]

    ax = plt.subplot(polar=True)
    cat_loc = np.linspace(start=0, stop=2 * np.pi, num=len(cat_linebreak))

    colors = ['#468dce', '#ffd044', '#d54141', '#ff8497', '#cb2fed']
    for idx, d in enumerate(data):
        ax.plot(cat_loc, d, 'o--', color=colors[idx], label=label[idx])
        ax.fill(cat_loc, d, alpha=0.15, color=colors[idx])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(cat_loc), cat_linebreak)

    for xtick, angle in zip(ax.get_xticklabels(), cat_loc):
    #     xtick.set_rotation(-angle * 180. / np.pi)
        if 0 < angle < np.pi:
            xtick.set_horizontalalignment('left')
        elif 0 < angle < np.pi*2:
            xtick.set_horizontalalignment('right')
        else:
            xtick.set_horizontalalignment('center')
        # xtick.set_rotation_mode("anchor")
    ax.xaxis.set_tick_params(labelsize=15.2)

    ax.set_ylim(0, 1.01)
    ax.set_rgrids([.2, .4, .6, .8, 1.])
    ax.invert_yaxis()

    ax.tick_params(axis='y', labelsize=8)
    ax.grid(color='#AAAAAA')
    ax.spines['polar'].set_color('#eaeaea')
    ax.set_facecolor('#FAFAFA')

    # ax.set_title(title, y=1.08)
    if len(data) > 1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.savefig(filename + '.png')
    plt.savefig(filename + '.svg')
    plt.savefig(filename + '.pdf')
    plt.show()
    plt.close()

def plot_traj_data(env, traj_data_history, plot_case, case_name, save_name, show_plot=False):
    """traj_data_history: (num_evaluate, NUM_CASE, nT, traj_dim)"""
    color_cycle_tab20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
                         '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                         '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
                         '#17becf', '#9edae5']

    variable_tag_lst = env.plot_info['variable_tag_lst']
    state_plot_idx_lst = env.plot_info['state_plot_idx_lst'] if 'state_plot_idx_lst' in env.plot_info else range(1, env.s_dim)
    ref_idx_lst = env.plot_info['ref_idx_lst']
    nrows_s, ncols_s = env.plot_info['state_plot_shape']
    nrows_a, ncols_a = env.plot_info['action_plot_shape']

    ref = env.ref_traj()
    x_axis = np.linspace(env.t0+env.dt, env.tT, num=env.nT)

    traj_mean = traj_data_history.mean(axis=0)
    traj_std = traj_data_history.std(axis=0)

    # State variables subplots
    fig1, ax1 = plt.subplots(nrows_s, ncols_s, figsize=(ncols_s*6, nrows_s*5))
    for i, fig_idx in enumerate(ref_idx_lst):
        ax1.flat[fig_idx-1].hlines(ref[i], env.t0, env.tT, color='r', linestyle='--', label='Set point')

    for fig_idx, i in enumerate(state_plot_idx_lst):
        ax1.flat[fig_idx].set_xlabel(variable_tag_lst[0])
        ax1.flat[fig_idx].set_ylabel(variable_tag_lst[fig_idx+1])
        if len(plot_case) > 2:
            ax1.flat[fig_idx].set_prop_cycle(color=color_cycle_tab20)
        for case in plot_case:
            ax1.flat[fig_idx].plot(x_axis, traj_mean[case, :, i], label=case_name[case])
            ax1.flat[fig_idx].fill_between(x_axis, traj_mean[case, :, i] + traj_std[case, :, i], traj_mean[case, :, i] - traj_std[case, :, i], alpha=0.5)
        ax1.flat[fig_idx].legend()
        ax1.flat[fig_idx].grid()
    fig1.tight_layout()
    plt.savefig(save_name + '_state_traj.png')
    plt.savefig(save_name + '_state_traj.svg')
    plt.savefig(save_name + '_state_traj.pdf')
    if show_plot:
        plt.show()
    plt.close()

    # Action variables subplots
    x_axis = np.linspace(env.t0, env.tT, num=env.nT)
    fig3, ax3 = plt.subplots(nrows_a, ncols_a, figsize=(ncols_a*6, nrows_a*5))
    for i in range(env.a_dim):
        axis = ax3.flat[i] if env.a_dim > 1 else ax3
        axis.set_xlabel(variable_tag_lst[0])
        axis.set_ylabel(variable_tag_lst[len(state_plot_idx_lst) + 1])
        if len(plot_case) > 2:
            axis.set_prop_cycle(color=color_cycle_tab20)
        for case in plot_case:
            axis.plot(x_axis, traj_mean[case, :, env.s_dim + i], label=case_name[case])
            axis.fill_between(x_axis, traj_mean[case, :, env.s_dim + i] + traj_std[case, :, env.s_dim + i], traj_mean[case, :, env.s_dim + i] - traj_std[case, :, env.s_dim + i], alpha=0.5)
        axis.legend()
        axis.grid()
    fig3.tight_layout()
    plt.savefig(save_name + '_action_traj.png')
    plt.savefig(save_name + '_action_traj.svg')
    plt.savefig(save_name + '_action_traj.pdf')
    if show_plot:
        plt.show()
    plt.close()

def plot_ref(env, traj_data_history, plot_case, case_name, save_name, show_plot=False):
    """traj_data_history: (num_evaluate, NUM_CASE, nT, traj_dim)"""
    color_cycle_tab20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
                         '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                         '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
                         '#17becf', '#9edae5']

    variable_tag_lst = env.plot_info['variable_tag_lst']
    state_plot_idx_lst = env.plot_info['state_plot_idx_lst'] if 'state_plot_idx_lst' in env.plot_info else range(1, env.s_dim)
    ref_idx_lst = env.plot_info['ref_idx_lst']

    ref = env.ref_traj()
    x_axis = np.linspace(env.t0+env.dt, env.tT, num=env.nT)

    traj_mean = traj_data_history.mean(axis=0)
    traj_std = traj_data_history.std(axis=0)

    fig1, ax1 = plt.subplots(1, 1, figsize=(7,4), layout='constrained')
    for i, ref_idx in enumerate(ref_idx_lst):
        ax1.hlines(ref[i], env.t0, env.tT, color='r', linestyle='--', label='Set point')

        ax1.set_xlabel(variable_tag_lst[0])
        ax1.set_ylabel(variable_tag_lst[ref_idx])
        if len(plot_case) > 2:
            ax1.set_prop_cycle(color=color_cycle_tab20)
        for case in plot_case:
            ax1.plot(x_axis, traj_mean[case, :, ref_idx], label=case_name[case])
            ax1.fill_between(x_axis, traj_mean[case, :, ref_idx] + traj_std[case, :, ref_idx],
                                     traj_mean[case, :, ref_idx] - traj_std[case, :, ref_idx], alpha=0.5)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 0.99))
        ax1.grid()
        ax1.set_xlim((env.t0, env.tT))
    # fig1.tight_layout()
    plt.savefig(save_name + '_target_state_traj.png')
    plt.savefig(save_name + '_target_state_traj.svg')
    plt.savefig(save_name + '_target_state_traj.pdf')
    if show_plot:
        plt.show()
    plt.close()
