import argparse
import torch
import numpy as np
import random

import algorithm
import environment


def get_config():
    parser = argparse.ArgumentParser(description='EPEL RL')

    # Basic settings
    parser.add_argument('--algo', type=str, default='QRDQN', help='RL algorithm')
    parser.add_argument('--env', type=str, default='PENICILLIN', help='Environment')
    parser.add_argument('--seed', type=int, default=0, help='Seed number')
    parser.add_argument('--device', type=str, default='cpu', help='Device - cuda or cpu')
    parser.add_argument('--save_freq', type=int, default=20, help='Save frequency')
    parser.add_argument('--save_model', action='store_true', help='Whether to save model or not')
    parser.add_argument('--load_model', action='store_true', help='Whether to load saved model or not')

    # Training settings
    parser.add_argument('--max_episode', type=int, default=200, help='Maximum training episodes')
    parser.add_argument('--init_ctrl_idx', type=int, default=0, help='Episodes for training with initial controller')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini-batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount')
    parser.add_argument('--warm_up_episode', type=int, default=10, help='Number of warm up episode')

    # Neural network parameters
    parser.add_argument('--num_hidden_nodes', type=int, default=128, help='Number of hidden nodes in MLP')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers in MLP')
    parser.add_argument('--tau', type=float, default=0.005, help='Parameter for soft target update')
    parser.add_argument('--adam_eps', type=float, default=1e-6, help='Epsilon for numerical stability')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='Weight decay (L2 penalty)')
    parser.add_argument('--grad_clip_mag', type=float, default=5.0, help='Gradient clipping magnitude')
    parser.add_argument('--critic_lr', type=float, default=1e-4, help='Critic network learning rate')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='Actor network learning rate')

    # RBF parameters
    parser.add_argument('--rbf_dim', type=int, default=10, help='Dimension of RBF basis function')
    parser.add_argument('--rbf_type', type=str, default='gaussian', help='Type of RBF basis function')

    args = parser.parse_args()

    # Algorithm specific settings
    if args.algo == 'A2C':
        args.use_mc_return = False
    elif args.algo == 'DDPG':
        pass
    elif args.algo == 'DQN':
        args.max_n_action_grid = 200
    elif args.algo == 'GDHP':
        args.costate_lr = 1e-4
    elif args.algo == 'iLQR':
        args.ilqr_alpha = 0.1
    elif args.algo == 'PI2':
        args.num_rollout = 5
        args.h = 10
        args.init_lambda = 100
    elif args.algo == 'PoWER':
        args.num_rollout = 10
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
        args.max_n_action_grid = 200
        args.n_quantiles = 21
    elif args.algo == 'REPS':
        args.max_kl_divergence = 0.01
        args.num_rollout = 2
        args.critic_reg = 0.01
        args.actor_reg = 0.01
        args.num_critic_update = 10
    elif args.algo == 'SAC':
        args.automatic_temp_tuning = True
        args.temperature = 0
    elif args.algo == 'SDDP':
        args.sddp_gamma = 0.5
    elif args.algo == 'TD3':
        args.policy_noise = 0.2
        args.noise_clip = 0.5
        args.policy_delay = 2
    elif args.algo == 'TRPO':
        args.gae_lambda = 0.99
        args.gae_gamma = 0.99
        args.num_critic_update = 5
        args.num_cg_iterations = 10
        args.num_line_search = 10
        args.max_kl_divergence = 0.01

    # Derivative setting
    args.need_derivs = False
    args.need_noise_derivs = False
    args.need_deriv_inverse = False

    if args.algo in ['GDHP', 'SDDP', 'iLQR']:
        args.need_derivs = True

        if args.algo == 'SDDP':
            args.need_noise_derivs = True
        elif args.algo == 'GDHP':
            args.need_deriv_inverse = True
    
    # Discrete action space setting
    args.is_discrete_action = False
    if args.algo in ['DQN', 'QRDQN']:
        args.is_discrete_action = True
       
    return args


def get_env(config):
    env_name = config.env

    if env_name == 'CSTR':
        env = environment.CSTR(config)
    elif env_name == 'POLYMER':
        env = environment.POLYMER(config)
    elif env_name == 'PENICILLIN':
        env = environment.PENICILLIN(config)
    elif env_name == 'CRYSTAL':
        env = environment.CRYSTAL(config)
    elif env_name == 'DISTILLATION':
        env = environment.DISTILLATION(config)
    elif env_name == 'PFR':
        env = environment.PfrEnv(config)
    else:
        raise NameError('Wrong environment name')

    return env


def get_algo(config, env):
    algo_name = config.algo
    config.s_dim = env.s_dim
    config.a_dim = env.a_dim
    config.p_dim = env.p_dim
    config.nT = env.nT
    config.dt = env.dt

    if algo_name == 'A2C':
        algo = algorithm.A2C(config)
    elif algo_name == 'DDPG':
        algo = algorithm.DDPG(config)
    elif algo_name == 'DQN':
        algo = algorithm.DQN(config)
    elif algo_name == 'GDHP':
        algo = algorithm.GDHP(config)
    elif algo_name == 'iLQR':
        algo = algorithm.iLQR(config)
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
    elif algo_name == 'TD3':
        algo = algorithm.TD3(config)
    elif algo_name == 'TRPO':
        algo = algorithm.TRPO(config)
    else:
        raise NameError('Wrong algorithm name')

    return algo


def set_seed(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
