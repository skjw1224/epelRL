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
    # available_algs = [alg.__name__ for alg in algorithm.__all__]
    # available_envs = [env.__name__ for env in environment.__all__]

    available_algs = ['TRPO']
    # available_algs = ['A2C', 'DDPG', 'DQN',
    #                   # 'iLQR', 'GDHP', 'SDDP',
    #                   'QRDQN', 'SAC', 'TD3',
    #                   'PPO', 'TRPO']
    #                   # 'PI2', 'PoWER', 'REPS']
    available_envs = ['POLYMER']  # POLYMER, 'CRYSTAL', 'DISTILLATION'
    # (critic_lr, adam_eps, l2_reg)
    hyp_params = {
        'A2C': (0.01, 1.e-6, 1.e-3, []),
        'DDPG': (0.001, 0.0001, 1.e-3, []),
        'DQN': (0.01, 1.e-6, 1.e-3, []),
        'iLQR': (1.e-4, 1.e-6, 1.e-3, []),
        'GDHP': (1.e-4, 1.e-6, 1.e-3, []),
        'SDDP': (1e-4, 1e-6, 1e-3, []),
        'QRDQN': (0.0001, 1.e-6, 0.1, []),
        'SAC': (1e-4, 1e-6, 1e-3, []),
        'TD3': (0.01, 1.e-6, 1.e-3, []),
        'PPO': (0.001, 1.e-6, 1.e-3, []),
        'TRPO': (1e-4, 1e-6, 1e-3, []),
        'PoWER': (1.e-4, 1.e-6, 1.e-3, 1000),
        'REPS': (1.e-4, 1.e-6, 1.e-3)
    }
    for env in available_envs:
        for idx, alg in enumerate(available_algs):
            lr, eps, l2reg, rbf_dim = hyp_params[alg]
            subprocess.run(['python', 'train_single_env_algo.py', '--algo', alg, '--env', env,
                            '--max_episode', "1000", '--save_freq', "100", '--warm_up_episode', '1',
                            '--convg_bound', '0.1', '--critic_lr', str(lr), '--adam_eps', str(eps),
                            '--l2_reg', str(l2reg), '--rbf_dim', str(rbf_dim)])
            subprocess.run(['python', 'test_single_env_algo.py', '--algo', alg, '--env', env,
                            '--rbf_dim', str(rbf_dim)])

        subprocess.run(['python', 'test_plot.py', '--env', env])
    subprocess.run(['python', 'performance_summary.py'])

if __name__ == '__main__':
    main()

