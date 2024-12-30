import subprocess
import os
import algorithm
import environment

def main():
    subprocess.run(['python', 'train_single_env_algo.py', '--algo', 'TRPO', '--env', 'CSTR',
                    '--max_episode', '1000', '--save_freq', '200'])
    # available_envs = [env.__name__ for env in environment.__all__]
    # for env in available_envs:
    #     subprocess.run(['python', 'test_plot.py', '--env', env])
    # subprocess.run(['python', 'performance_summary.py'])

def main_RUN():
    # available_algs = [alg.__name__ for alg in algorithm.__all__]
    # available_envs = [env.__name__ for env in environment.__all__]
    available_algs = ['iLQR', 'SDDP']
    available_envs = ['CSTR']

    for env in available_envs:
        for idx, alg in enumerate(available_algs):
            subprocess.run(['python', 'train_single_env_algo.py', '--algo', alg, '--env', env])
            subprocess.run(['python', 'test_single_env_algo.py', '--algo', alg, '--env', env])

        subprocess.run(['python', 'test_plot.py', '--env', env])
    subprocess.run(['python', 'performance_summary.py'])

if __name__ == '__main__':
    main()

