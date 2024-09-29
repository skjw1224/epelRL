import subprocess
import os
import algorithm
import environment

def main():
    # available_algs = [alg.__name__ for alg in algorithm.__all__]
    # available_envs = [env.__name__ for env in environment.__all__]

    available_algs = ['A2C', 'DDPG']
    available_envs = ['CSTR']
    for env in available_envs:
        for alg in available_algs:
            # subprocess.run(['python', 'train_single_env_algo.py', '--algo', alg, '--env', env,
            #                 '--max_episode', "5", '--save_freq', "2", '--warm_up_episode', '50'])
            subprocess.run(['python', 'train_and_test_single_env_algo.py', '--algo', alg, '--env', env,
                            '--num_test_evaluate', "10"])


if __name__ == '__main__':
    main()

