import subprocess
import os
import algorithm
import environment

def main():
    available_alg = [alg.__name__ for alg in algorithm.__all__]
    available_envs = [env.__name__ for env in environment.__all__]
    for alg in available_alg:
        for env in available_envs:
            subprocess.run(['python', 'train_single_env_algo.py', '--algo', alg, '--env', env, '--max_episode', "10", '--save_freq', "2"])

if __name__ == '__main__':
    main()

