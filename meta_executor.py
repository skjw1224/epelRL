import subprocess
import os
import algorithm
import environment

def main_tuning():
    # alg = 'QRDQN'
    env = 'CSTR'
    available_algs = ['QRDQN', 'TRPO']
    lr_algs = [0.0001, 0.0001]
    value_lst = [0.1, 0.01, 0.0001, 0.00001]

    for idx, alg in enumerate(available_algs):
        for v in value_lst:
            lr = lr_algs[idx]
            subprocess.run(['python', 'train_single_env_algo.py', '--algo', alg, '--env', env,
                            '--max_episode', "100", '--save_freq', "20", '--warm_up_episode', '1',
                            '--convg_bound', '0.01', '--critic_lr', str(lr), '--l2_reg', str(v)])


def main():
    # available_algs = [alg.__name__ for alg in algorithm.__all__]
    # available_envs = [env.__name__ for env in environment.__all__]

    available_algs = ['A2C', 'DDPG']
    available_envs = ['CSTR']
    for env in available_envs:
        for alg in available_algs:
            subprocess.run(['python', 'train_single_env_algo.py', '--algo', alg, '--env', env,
                            '--max_episode', "5", '--save_freq', "1", '--warm_up_episode', '1',
                            '--convg_bound', '2.5e-2'])
            subprocess.run(['python', 'test_single_env_algo.py', '--algo', alg, '--env', env])

        subprocess.run(['python', 'test_plot.py', '--env', env])
    subprocess.run(['python', 'performance_summary.py'])

if __name__ == '__main__':
    main()

