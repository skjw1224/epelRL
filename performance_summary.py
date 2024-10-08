import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import algorithm

def performance_summary():
    # available_algs = [alg.__name__ for alg in algorithm.__all__]
    # available_envs = [env.__name__ for env in environment.__all__]
    # available_algs = ['A2C', 'DDPG', 'DQN', 'iLQR', 'PPO', 'QRDQN', 'SAC', 'TD3', 'SDDP']
    available_envs = ['CRYSTAL', 'CSTR']
    env_name = available_envs[0]
    available_file_path = glob.glob(f'./_Result/test_{env_name}_*')
    available_algs = []
    for dir_path in available_file_path:
        dir_name = os.path.basename(dir_path)
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
        history.append([avg_summary[i] for i in range(len(avg_summary))])
        alg_df = pd.DataFrame(np.concatenate((alg_summary, avg_summary.reshape([1,-1]))), columns=summary_feature,
                              index=available_envs+['Average'])
        alg_df.to_csv(os.path.join(summary_path, f'table_{alg_name}.csv'))

    # Bar Chart
    for f, feat_name in enumerate(summary_feature):
        fig_bar, ax_bar = plt.subplots()
        ax_bar.grid(color='grey', linestyle=':', zorder=0)
        bars = ax_bar.bar(available_algs, [h[f] for h in history], zorder=3)
        if f==1:
            ax_bar.bar_label(bars, padding=1)
        else:
            ax_bar.bar_label(bars, padding=1, fmt='%.2f')
        ax_bar.set_title(feat_name)
        plt.savefig(os.path.join(summary_path, f'bar_{feat_name}.png'))
        plt.show()
        plt.close()

    # Radar Chart
    scaling_factor = {"min": [min([h[f] for h in history]) for f in range(len(summary_feature))],
                      "max": [max([h[f] for h in history]) for f in range(len(summary_feature))]}
    scaling_factor["scale"] = [max(scaling_factor["max"][f] - scaling_factor["min"][f], 0.01)for f in range(len(summary_feature))]

    cat = summary_feature
    cat = [*cat, cat[0]]
    radar_data = {}

    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(cat))

    for i, alg_name in enumerate(available_algs):
        ax_alg = plt.subplot(polar=True)

        alg_data = [(history[i][f] - scaling_factor["min"][f]) / scaling_factor["scale"][f]
                    for f in range(len(summary_feature))]
        alg_data = [*alg_data, alg_data[0]]
        radar_data[alg_name] = alg_data

        ax_alg.plot(label_loc, alg_data, 'o--', color='#468dce')
        ax_alg.fill(label_loc, alg_data, alpha=0.15, color='#468dce')
        ax_alg.set_theta_offset(np.pi / 2)
        ax_alg.set_theta_direction(-1)
        ax_alg.set_thetagrids(np.degrees(label_loc), cat)

        for label, angle in zip(ax_alg.get_xticklabels(), label_loc):
            label.set_rotation(angle * 180. / np.pi - 90.)
            if 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')
            label.set_rotation_mode("anchor")

        # Ensure radar goes from 0 to 100. it also removes the extra line
        ax_alg.set_ylim(0, 1.01)
        # You can also set gridlines manually like this:
        ax_alg.set_rgrids([.2, .4, .6, .8, 1.])
        ax_alg.invert_yaxis()

        # Set position of y-labels (0-100) to be in the middle
        # of the first two axes.
        # ax.set_rlabel_position(180 / n_points)

        # Make the y-axis (0-100) labels smaller.
        ax_alg.tick_params(axis='y', labelsize=8)
        # Change the color of the circular gridlines.
        ax_alg.grid(color='#AAAAAA')
        # Change the color of the outermost gridline (the spine).
        ax_alg.spines['polar'].set_color('#eaeaea')
        # Change the background color inside the circle itself.
        ax_alg.set_facecolor('#FAFAFA')

        # Add title.
        ax_alg.set_title(alg_name, y=1.08)

        # Add a legend as well.
        # ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.savefig(os.path.join(summary_path, f'{alg_name}.png'))
        plt.show()
        plt.close()

    # Radar Chart per Group
    alg_groups = {'Value-based method': ['DQN', 'QRDQN'],
                  'Policy-based method': ['A2C', 'TRPO', 'PPO', 'DDPG', 'TD3'],
                  'Inference-based method': ['PoWER', 'REPS', 'PI2', 'SAC'],
                  'Model-based method': ['iLQR', 'SDDP', 'GDHP']}
    colors = ['#468dce', '#ffd044', '#d54141', '#ff8497', '#cb2fed']

    for key, val in alg_groups.items():
        ax_group = plt.subplot(polar=True)
        for i, alg_name in enumerate(val):
            if alg_name in radar_data.keys():
                alg_data = radar_data[alg_name]
                ax_group.plot(label_loc, alg_data, 'o--', color=colors[i], label=alg_name)
                ax_group.fill(label_loc, alg_data, alpha=0.15, color=colors[i])
        ax_group.set_theta_offset(np.pi / 2)
        ax_group.set_theta_direction(-1)
        ax_group.set_thetagrids(np.degrees(label_loc), cat)

        for label, angle in zip(ax_group.get_xticklabels(), label_loc):
            if 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

        ax_group.set_ylim(0, 1.01)
        ax_group.set_rgrids([.2, .4, .6, .8, 1.])
        ax_group.invert_yaxis()

        ax_group.tick_params(axis='y', labelsize=8)
        ax_group.grid(color='#AAAAAA')
        ax_group.spines['polar'].set_color('#eaeaea')
        ax_group.set_facecolor('#FAFAFA')

        ax_group.set_title(key, y=1.08)
        ax_group.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.savefig(os.path.join(summary_path, f'radar_{key}.png'))
        plt.show()
        plt.close()

if __name__ == '__main__':
    performance_summary()
