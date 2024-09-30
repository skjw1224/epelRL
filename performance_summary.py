import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import algorithm

def performance_summary():
    # available_algs = [alg.__name__ for alg in algorithm.__all__]
    # available_envs = [env.__name__ for env in environment.__all__]
    available_algs = ['A2C', 'DDPG']
    available_envs = ['CSTR']

    history = []
    summary_feature = ['Episodic Training Computation Time', 'Number of Training Episodes',
                       'Convergence Criteria at Training Termination',
                       'Test Performance Average', 'Test Performance STD']
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

    cat = summary_feature
    cat = [*cat, cat[0]]

    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(cat))
    summary_path = os.path.join(f'./_Result', 'summary')
    os.makedirs(summary_path, exist_ok=True)

    for i, alg_name in enumerate(available_algs):
        alg_ax = plt.subplot(polar=True)

        alg_data = history[i]
        alg_data = [*alg_data, alg_data[0]]

        alg_ax.plot(label_loc, alg_data, 'o--', color='blue')
        alg_ax.fill(label_loc, alg_data, alpha=0.15, color='blue')
        alg_ax.set_theta_offset(np.pi / 2)
        alg_ax.set_theta_direction(-1)
        alg_ax.set_thetagrids(np.degrees(label_loc), cat)

        for label, angle in zip(alg_ax.get_xticklabels(), label_loc):
            if 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

        # Ensure radar goes from 0 to 100. it also removes the extra line
        alg_ax.set_ylim(0, 5)
        # You can also set gridlines manually like this:
        # ax.set_rgrids([20, 40, 60, 80, 100])

        # Set position of y-labels (0-100) to be in the middle
        # of the first two axes.
        # ax.set_rlabel_position(180 / n_points)

        # Make the y-axis (0-100) labels smaller.
        alg_ax.tick_params(axis='y', labelsize=8)
        # Change the color of the circular gridlines.
        alg_ax.grid(color='#AAAAAA')
        # Change the color of the outermost gridline (the spine).
        alg_ax.spines['polar'].set_color('#eaeaea')
        # Change the background color inside the circle itself.
        alg_ax.set_facecolor('#FAFAFA')

        # Add title.
        alg_ax.set_title(alg_name, y=1.08)

        # Add a legend as well.
        # ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.savefig(os.path.join(summary_path, f'{alg_name}.png'))
        plt.show()
        plt.close()

if __name__ == '__main__':
    performance_summary()
