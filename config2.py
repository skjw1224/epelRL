import argparse
import torch
import numpy as np
import random

import algorithm
import environment


def get_config():
    parser = argparse.ArgumentParser(description='EPEL RL')

    # Basic settings
    parser.add_argument('--algo', type=str, default='DQN', help='RL algorithm')
    parser.add_argument('--env', type=str, default='CSTR', help='Environment')
    parser.add_argument('--seed', type=int, default=0, help='Seed number')
    parser.add_argument('--device', type=str, default='cuda', help='Device - cuda or cpu')
    parser.add_argument('--model_save_freq', type=int, default=10, help='Model save frequency')
    parser.add_argument('--model_save_path', type=str, default='./_models/', help='Model save path')
    parser.add_argument('--result_save_path', type=str, default='./_results/', help='Result save path')
    parser.add_argument('--save_model', action='store_true', help='Whether to save model or not')
    parser.add_argument('--load_model', action='store_false', help='Whether to load saved model or not')

    # Training settings
    parser.add_argument('--max_episode', type=int, default=100, help='Maximum training episodes')
    parser.add_argument('--init_ctrl_idx', type=int, default=0, help='Episodes for training with initial controller')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini-batch size')

    # Neural network parameters
    parser.add_argument('--num_hidden_nodes', type=int, default=128, help='Number of hidden nodes in MLP')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers in MLP')
    parser.add_argument('--tau', type=float, default=0.005, help='Parameter for soft target update')
    parser.add_argument('--adam_eps', type=float, default=1e-6, help='Epsilon for numerical stability')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='Weight decay (L2 penalty)')
    parser.add_argument('--grad_clip_mag', type=float, default=5.0, help='Gradient clipping magnitude')
    parser.add_argument('--critic_lr', type=float, default=1e-5, help='Critic network learning rate')
    parser.add_argument('--actor_lr', type=float, default=1e-5, help='Actor network learning rate')

    # RBF parameters
    parser.add_argument('--rbf_dim', type=int, default=10, help='Dimension of RBF basis function')
    parser.add_argument('--rbf_type', type=str, default='gaussian', help='Type of RBF basis function')

    args = parser.parse_args()

    # Algorithm specific settings
    if args.algo == 'A2C':
        pass
    elif args.algo == 'DDPG':
        pass
    elif args.algo == 'DQN':
        args.single_dim_mesh = [-1., -.9, -.5, -.2, -.1, -.05, 0., .05, .1, .2, .5, .9, 1.]
    elif args.algo == 'GDHP':
        args.costate_lr = 1e-4
    elif args.algo == 'iLQR':
        pass
    elif args.algo == 'PI2':
        args.num_rollout = 5
        args.h = 10
        args.init_lambda = 100
    elif args.algo == 'PoWER':
        args.batch_epi = 10
        args.variance_update = True
    elif args.algo == 'PPO':
        args.gae_lambda = 0.99
        args.gae_gamma = 0.99
        args.num_critic_update = 5
        args.num_cg_iterations = 10
        args.num_line_search = 10
        args.max_kl_divergence = 0.01
        args.clip_epsilon = 0.1
    elif args.algo == 'QRDQN':
        args.n_quantiles = 21
    elif args.algo == 'REPS':
        args.max_kl_divergence = 0.01
        args.batch_epi = 2
        args.critic_reg = 0.01
        args.actor_reg = 0.01
        args.num_critic_update = 10
    elif args.algo == 'SAC':
        args.automatic_temp_tuning = True
        args.temperature = 0
    elif args.algo == 'SDDP':
        pass
    elif args.algo == 'TRPO':
        args.gae_lambda = 0.99
        args.gae_gamma = 0.99
        args.num_critic_update = 5
        args.num_cg_iterations = 10
        args.num_line_search = 10
        args.max_kl_divergence = 0.01
        args.clip_epsilon = 0.1

    return args


def get_env(config):
    env_name = config.env

    if env_name == 'CSTR':
        env = environment.CSTR()
    else:
        raise NameError('Wrong environment name')

    return env


def get_algo(config, env):
    algo_name = config.algo

    config.s_dim = env.s_dim
    config.a_dim = env.a_dim
    config.nT = env.nT

    if algo_name in ['DQN', 'QRDQN', 'DDPG', 'SAC', 'GDHP']:
        config.update_type = 'single_train_per_single_step'
    elif algo_name in ['A2C', 'TRPO', 'PPO', 'iLQR', 'SDDP']:
        config.update_type = 'single_train_per_single_episode'
    elif algo_name in ['REPS', 'PoWER', 'PI2']:
        config.update_type = 'single_train_per_multiple_episodes'
    else:
        raise NameError('Wrong algorithm name')

    if algo_name == 'A2C':
        algo = algorithm.A2C(config)
    elif algo_name == 'DDPG':
        algo = algorithm.DDPG(config)
    elif algo_name == 'DQN':
        algo = algorithm.DQN(config)
    elif algo_name == 'GDHP':
        algo = algorithm.GDHP(config)
    elif algo_name == 'iLQR':
        algo = algorithm.iLQR(config, env)
    elif algo_name == 'PI2':
        algo = algorithm.PI2(config)
    elif algo_name == 'PoWER':
        algo = algorithm.PoWER(config)
    elif algo_name == 'PPO':
        algo = algorithm.PPO(config)
    elif algo_name == 'QRDQN':
        algo = algorithm.QRDQN(config)
    elif algo_name == 'REPS':
        algo = algorithm.REPS(config)
    elif algo_name == 'SAC':
        algo = algorithm.SAC(config)
    elif algo_name == 'SDDP':
        algo = algorithm.SDDP(config)
    elif algo_name == 'TRPO':
        algo = algorithm.TRPO(config)
    else:
        raise NameError('Wrong algorithm name')

    return algo


def set_seed(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
