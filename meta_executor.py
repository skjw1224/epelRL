import subprocess
import os
import algorithm
import environment

def main():
    # available_algs = [alg.__name__ for alg in algorithm.__all__]
    # available_envs = [env.__name__ for env in environment.__all__]

    available_algs = ['DDPG']
    available_envs = ['CSTR']
    for alg in available_algs:
        for env in available_envs:
            subprocess.run(['python', 'train_single_env_algo.py', '--algo', alg, '--env', env,
                            '--max_episode', "20", '--save_freq', "5"])

if __name__ == '__main__':
    main()

