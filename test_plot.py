import os
import glob
from config import get_config, get_env, plot_traj_data
import numpy as np

def test_plot():
    # Basic configurations
    config = get_config()
    env_name = config['env']
    print(f'Test Summary in {env_name}')
    available_file_path = glob.glob(f'./_Result/test_{env_name}_*')
    available_alg_names = []
    for dir_path in available_file_path:
        dir_name = os.path.basename(dir_path)
        alg_name = dir_name[6 + len(env_name):]
        available_alg_names.append(alg_name)

    # Set save path
    config['save_path'] = os.path.join(os.getcwd(), '_Result', f'test_all_{env_name}')
    os.makedirs(config['save_path'], exist_ok=True)

    env = get_env(config)

    traj_data_history = []
    cost_traj_data_history = []

    for alg_name in available_alg_names:
        dir_path = os.path.join(f'./_Result/test_{env_name}_{alg_name}')
        traj = np.load(os.path.join(dir_path, 'test_traj_data_history.npy'))
        cost_traj = np.load(os.path.join(dir_path, 'test_cost_history.npy'))

        traj_data_history.append(traj)
        cost_traj_data_history.append(cost_traj)
        avg_cost, std_cost = np.average(cost_traj), np.std(cost_traj)
        print(f'{alg_name}: Avg cost {avg_cost:.4f} || Std cost {std_cost:.4f}')
    traj_data_history = np.concatenate(traj_data_history, axis=1)

    save_name = os.path.join(config['save_path'], 'traj')
    case_plot = [i for i in range(len(available_alg_names))]
    plot_traj_data(env, traj_data_history, case_plot, available_alg_names, save_name)


if __name__ == '__main__':
    test_plot()