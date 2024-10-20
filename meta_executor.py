import subprocess
import os
import algorithm
import environment

def main_TUNING():
    # alg = 'QRDQN'
    env = 'POLYMER'
    available_algs = ['DQN']

    for idx, alg in enumerate(available_algs):
        subprocess.run(['python', 'train_single_env_algo.py', '--algo', alg, '--env', env,
                        '--max_episode', "10", '--save_freq', "2", '--warm_up_episode', '1',
                        '--convg_bound', '0.01', '--rbf_dim', '100', '--rbf_type', 'matern32'])
        # subprocess.run(['python', 'test_single_env_algo.py', '--algo', alg, '--env', env])

    # subprocess.run(['python', 'train_single_env_algo.py', '--algo', 'REPS', '--env', env,
    #                 '--max_episode', "10", '--save_freq', "2", '--warm_up_episode', '1'])
    # subprocess.run(['python', 'test_single_env_algo.py', '--algo', 'REPS', '--env', env])


def main():
    available_algs = [alg.__name__ for alg in algorithm.__all__]
    available_envs = [env.__name__ for env in environment.__all__]

    for env in available_envs:
        for idx, alg in enumerate(available_algs):
            subprocess.run(['python', 'train_single_env_algo.py', '--algo', alg, '--env', env])
            subprocess.run(['python', 'test_single_env_algo.py', '--algo', alg, '--env', env])

        subprocess.run(['python', 'test_plot.py', '--env', env])
    subprocess.run(['python', 'performance_summary.py'])

if __name__ == '__main__':
    main()

