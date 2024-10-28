import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import algorithm
from utility.plotting import plot_radar

alg_groups = {'Value-based method': ['DQN', 'QRDQN'],
              'Policy-based method': ['A2C', 'TRPO', 'PPO', 'DDPG', 'TD3'],
              'Inference-based method': ['PoWER', 'REPS', 'PI2', 'SAC'],
              'Model-based method': ['iLQR', 'SDDP', 'GDHP']}

def get_summary_history():
    # available_algs = [alg.__name__ for alg in algorithm.__all__]
    # available_envs = [env.__name__ for env in environment.__all__]
    # available_algs = ['A2C', 'DDPG', 'DQN', 'iLQR', 'PPO', 'QRDQN', 'SAC', 'TD3', 'SDDP']

    available_envs = ['DISTILLATION', 'POLYMER', 'PFR', 'PENICILLIN', 'CSTR', 'CRYSTAL']
    env_name = available_envs[0]
    available_file_path = glob.glob(f'./_Result/test_{env_name}_*')
    available_algs = []
    for dir_path in available_file_path:
        dir_name = os.path.basename(dir_path)
        if os.path.isfile(os.path.join(dir_path, 'test_traj_data_history.npy')):
            alg_name = dir_name[6 + len(env_name):]
            available_algs.append(alg_name)
    summary_path = os.path.join(f'./_Result', 'summary')
    os.makedirs(summary_path, exist_ok=True)

    history = []
    summary_feature = ['Episodic Computation', 'Converged Epi',
                       'Convergence Criteria',
                       'Performance Average', 'Performance STD']
    for alg_name in available_algs:
        alg_summary = np.zeros((len(available_envs), len(summary_feature)))
        for i, env_name in enumerate(available_envs):
            test_path = os.path.join(f'./_Result/test_{env_name}_{alg_name}')
            cost_traj = np.load(os.path.join(test_path, 'test_cost_history.npy'))

            train_path = os.path.join(f'./_Result/{env_name}_{alg_name}')
            train_stat = np.load(os.path.join(train_path, 'learning_stat_history.npy'))
            train_episodic_computation_time = np.load(os.path.join(train_path, 'computation_time.npy'))

            train_termination_episode = len(train_stat[:, 0])
            train_termination_convg_criteria = train_stat[-1, 1]

            test_performance_mean = np.mean(cost_traj)
            test_performance_std = np.std(cost_traj)

            alg_summary[i, :] = [train_episodic_computation_time, train_termination_episode,
                                 train_termination_convg_criteria,
                                 test_performance_mean, test_performance_std]
        avg_summary = np.mean(alg_summary, axis=0)
        # err_minus = np.std(alg_summary, axis=0)
        # err_plus = np.std(alg_summary, axis=0)
        err_minus = avg_summary - np.min(alg_summary, axis=0)
        err_plus = np.max(alg_summary, axis=0) - avg_summary
        median_summary = np.median(alg_summary, axis=0)

        history.append([(avg_summary[i], err_minus[i], err_plus[i], median_summary[i]) for i in range(len(avg_summary))])

        alg_df = pd.DataFrame(np.concatenate((alg_summary, avg_summary.reshape([1,-1]),
                                              err_minus.reshape([1,-1]), err_plus.reshape([1,-1]),
                                              median_summary.reshape([1,-1]))),
                              columns=summary_feature,
                              index=available_envs+['Average', 'Minus Err', 'Plus Err', 'Median'])
        alg_df.to_csv(os.path.join(summary_path, f'table_{alg_name}.csv'))

    return available_algs, available_envs, summary_path, summary_feature, history

def plot_per_algorithm():
    available_algs, available_envs, summary_path, summary_feature, history = get_summary_history()

    # Bar Chart
    bar_width = 0.25
    ftsize = 12
    x_loc = np.arange(len(available_algs))
    for f, feat_name in enumerate(summary_feature):
        fig_bar, ax_bar = plt.subplots(figsize=(10,6))
        ax_bar.grid(color='grey', linestyle=':', zorder=0)
        bars = ax_bar.bar(x_loc, [h[f][0] for h in history],
                          width=bar_width, label='Average',
                          # yerr=[[h[f][1] for h in history], [h[f][2] for h in history]],
                          zorder=3)
        bars2 = ax_bar.bar(x_loc+bar_width, [h[f][-1] for h in history],
                           width=bar_width, label='Median',
                           zorder=3)
        ax_bar.set_xticks(x_loc, available_algs)
        ax_bar.tick_params(axis='x', labelsize=ftsize)
        ax_bar.tick_params(axis='y', labelsize=ftsize)
        # if f==1:
        #     ax_bar.bar_label(bars, padding=1)
        #     ax_bar.bar_label(bars2, padding=1)
        # else:
        #     ax_bar.bar_label(bars, padding=1, fmt='%.2f')
        #     ax_bar.bar_label(bars2, padding=1, fmt='%.2f')
        ax_bar.legend(loc='center right', fontsize=ftsize-1)
        ax_bar.set_title(feat_name, fontsize=ftsize+5)
        plt.savefig(os.path.join(summary_path, f'bar_{feat_name}.png'))
        plt.show()
        plt.close()

    # Radar Chart
    scaling_factor = {"min": [min([h[f][0] for h in history]) for f in range(len(summary_feature))],
                      "max": [max([h[f][0] for h in history]) for f in range(len(summary_feature))]}
    scaling_factor["scale"] = [max(scaling_factor["max"][f] - scaling_factor["min"][f], 0.01)for f in range(len(summary_feature))]

    radar_data = {}
    for i, alg_name in enumerate(available_algs):
        alg_data = [(history[i][f][0] - scaling_factor["min"][f]) / scaling_factor["scale"][f]
                    for f in range(len(summary_feature))]
        radar_data[alg_name] = alg_data
        alg_filename = os.path.join(summary_path, f'{alg_name}.png')
        plot_radar([alg_data], [alg_name], summary_feature, alg_filename, alg_name)

    # Radar Chart per Group
    for key, val in alg_groups.items():
        group_data = []
        valid_algo = []
        for i, alg_name in enumerate(val):
            if alg_name in radar_data.keys():
                group_data.append(radar_data[alg_name])
                valid_algo.append(alg_name)
        filename = os.path.join(summary_path, f'radar_{key}.png')
        plot_radar(group_data, valid_algo, summary_feature, filename, key)

def plot_per_env():
    available_algs, available_envs, summary_path, summary_feature, _ = get_summary_history()

    env_groups = {'Regulation': ['CSTR', 'DISTILLATION', 'PFR'],
                 'Maximization': ['CRYSTAL', 'PENICILLIN', 'POLYMER']}
    alg_summary = {}
    for alg_name in available_algs:
        alg_summary[alg_name] = {}
        alg_df = pd.read_csv(os.path.join(summary_path, f'table_{alg_name}.csv'))
        for env_group_name, env_group_elements in env_groups.items():
            env_group_data = []
            for env in env_group_elements:
                env_data = alg_df[alg_df['Unnamed: 0'] == env].to_numpy()
                if len(env_data)>0:
                    env_group_data.append(env_data[:,1:].reshape([-1]))
            alg_summary[alg_name][env_group_name] = np.average(np.array(env_group_data), axis=0).tolist()
    scaling_factor = {"min": {}, "max": {}, "scale": {}}
    for env_group_name in env_groups.keys():
        scaling_factor["min"][env_group_name] = [min([h[env_group_name][f] for h in alg_summary.values()])
                                                 for f in range(len(summary_feature))]
        scaling_factor["max"][env_group_name] = [max([h[env_group_name][f] for h in alg_summary.values()])
                                                 for f in range(len(summary_feature))]
        scaling_factor["scale"][env_group_name] = \
            [max(scaling_factor["max"][env_group_name][f] - scaling_factor["min"][env_group_name][f], 0.01)
             for f in range(len(summary_feature))]

    for alg_group_name, alg_group_elements in alg_groups.items():
        group_data = []
        for alg in alg_group_elements:
            if alg in alg_summary.keys():
                group_data.append(alg_summary[alg])
        env_group_data = []
        env_group_label = []
        for env_group_idx, env_group_name in enumerate(env_groups.keys()):
            avg_val = np.average(np.array([g[env_group_name] for g in group_data]), axis=0).tolist()
            avg_val_scaled = [(avg_val[i] - scaling_factor["min"][env_group_name][i]) / scaling_factor["scale"][env_group_name][i] for i in range(len(summary_feature))]
            env_group_data.append([*avg_val_scaled])
            env_group_label.append(env_group_name)
        filename = os.path.join(summary_path, f'env_radar_{alg_group_name}.png')

        plot_radar(env_group_data, env_group_label, summary_feature, filename, alg_group_name)

    alg_list = []
    val_list = {x: [[] for _ in env_groups.keys()] for x in summary_feature}
    for alg_group_name, alg_group_elements in alg_groups.items():
        alg_list.extend(alg_group_elements)
        for alg in alg_group_elements:
            for env_group_idx, env_group_name in enumerate(env_groups.keys()):
                feat_val = alg_summary[alg][env_group_name]
                feat_scaled = [(feat_val[i] - scaling_factor["min"][env_group_name][i])/scaling_factor["scale"][env_group_name][i] for i in range(len(summary_feature))]
                for idx, f in enumerate(summary_feature):
                    val_list[f][env_group_idx].append(feat_scaled[idx])
    for f in summary_feature:
        plot_radar(val_list[f], env_group_label, alg_list,
                   os.path.join(summary_path, f'env_radar_{f}.png'), str(f))


if __name__ == '__main__':
    plot_per_algorithm()
    plot_per_env()
