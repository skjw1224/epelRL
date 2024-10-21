import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from config import get_config, get_env, plot_traj_data

def sorting_rule(e):
    return e[1]

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

    fig1, ax1 = plt.subplots(1, 1)
    for i, ref_idx in enumerate(ref_idx_lst):
        ax1.hlines(ref[i], env.t0, env.tT, color='r', linestyle='--', label='Set point')

        ax1.set_xlabel(variable_tag_lst[0])
        ax1.set_ylabel(variable_tag_lst[ref_idx])
        if len(plot_case) > 2:
            ax1.set_prop_cycle(color=color_cycle_tab20)
        for case in plot_case:
            ax1.plot(x_axis, traj_mean[case, :, ref_idx-1], label=case_name[case])
            ax1.fill_between(x_axis, traj_mean[case, :, ref_idx-1] + traj_std[case, :, ref_idx-1],
                                     traj_mean[case, :, ref_idx-1] - traj_std[case, :, ref_idx-1], alpha=0.5)
        ax1.legend(loc='upper left')
        ax1.grid()
    fig1.tight_layout()
    plt.savefig(save_name + '_target_state_traj.png')
    if show_plot:
        plt.show()
    plt.close()

def test_plot():
    # Basic configurations
    config = get_config()
    env_name = config['env']
    print(f'Test Summary in {env_name}')
    available_file_path = glob.glob(f'./_Result/test_{env_name}_*')
    available_alg_names = []
    for dir_path in available_file_path:
        dir_name = os.path.basename(dir_path)
        if os.path.isfile(os.path.join(dir_path, 'test_traj_data_history.npy')):
            alg_name = dir_name[6 + len(env_name):]
            available_alg_names.append(alg_name)

    # Set save path
    config['save_path'] = os.path.join(os.getcwd(), '_Result', f'test_all_{env_name}')
    os.makedirs(config['save_path'], exist_ok=True)
    show_plot = config['show_plot']

    env = get_env(config)

    traj_data_history = []
    cost_traj_data_history = []
    train_stat_data_history = []
    summary_history = []

    for alg_name in available_alg_names:
        test_path = os.path.join(f'./_Result/test_{env_name}_{alg_name}')
        traj = np.load(os.path.join(test_path, 'test_traj_data_history.npy'))
        cost_traj = np.load(os.path.join(test_path, 'test_cost_history.npy'))

        train_path = os.path.join(f'./_Result/{env_name}_{alg_name}')
        train_stat = np.load(os.path.join(train_path, 'learning_stat_history.npy'))

        traj_data_history.append(traj)
        cost_traj_data_history.append(cost_traj)
        train_stat_data_history.append(train_stat[:, :2])   # only cost and convergence criteria

        avg_cost, std_cost = np.average(cost_traj), np.std(cost_traj)
        convg_epi, convg_criteria = len(train_stat), train_stat[-1,1]
        summary_history.append([alg_name, avg_cost, std_cost, convg_epi, convg_criteria])
    traj_data_history = np.concatenate(traj_data_history, axis=1)

    save_name = os.path.join(config['save_path'], 'traj')
    plot_case = [i for i in range(len(available_alg_names))]
    plot_traj_data(env, traj_data_history, plot_case, available_alg_names, save_name, show_plot)
    plot_ref(env, traj_data_history, plot_case, available_alg_names, save_name, show_plot)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
    stat_lst = ['Cost', 'Convergence criteria']
    for i, stat_name in enumerate(stat_lst):
        ax.flat[i].set_xlabel('Episode')
        ax.flat[i].set_ylabel(stat_name)
        ax.flat[i].set_prop_cycle(color=[
            '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
            '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
            '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
            '#17becf', '#9edae5'])
        for case in plot_case:
            ax.flat[i].plot(train_stat_data_history[case][:,i], label=available_alg_names[case])
        ax.flat[i].legend()
        ax.flat[i].grid()
    fig.tight_layout()
    plt.savefig(save_name + '_train_stat.png')
    if show_plot:
        plt.show()
    plt.close()

    summary_history.sort(key=sorting_rule)
    print('Test cost average || Test cost std || Termination epi || Termination criteria')
    for summary in summary_history:
        alg_name, avg_cost, std_cost, convg_epi, convg_criteria = summary
        print(f'{alg_name}: {avg_cost:.4f} || {std_cost:.4f} || {convg_epi} || {convg_criteria:.8f}')


if __name__ == '__main__':
    test_plot()
