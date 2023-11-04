import argparse
import torch
import numpy as np
import random

from algorithm import a2c, ddpg, dqn, gdhp, ilqr, pi2, power, ppo, qrdqn, reps, sac, sddp, trpo
from environment import env_casadi


def get_config():
    parser = argparse.ArgumentParser(description='EPEL RL')

    # Basic settings
    parser.add_argument('--algo', type=str, default='SAC', help='RL algorithm')
    parser.add_argument('--env', type=str, default='CSTR', help='Environment')
    parser.add_argument('--seed', type=int, default=0, help='Seed number')
    parser.add_argument('--save_model', type=bool, default=True, help='')
    parser.add_argument('--load_model', type=bool, default=False, help='')
    parser.add_argument('--model_save_freq', type=int, default=10, help='')
    parser.add_argument('--model_save_path', type=str, default='./_models/', help='')
    parser.add_argument('--result_save_path', type=str, default='./_results/', help='')

    # Training settings
    parser.add_argument('--max_episode', type=int, default=10, help='')
    parser.add_argument('--init_ctrl_idx', type=int, default=0, help='')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')

    # Neural network parameters
    parser.add_argument('--num_hidden_nodes', type=int, default=128, help='')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='')
    parser.add_argument('--tau', type=float, default=0.005, help='')
    parser.add_argument('--adam_eps', type=float, default=1e-4, help='')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='')
    parser.add_argument('--grad_clip_mag', type=float, default=5.0, help='')
    parser.add_argument('--critic_lr', type=float, default=1e-3, help='')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='')

    # RBF parameters
    parser.add_argument('--rbf_dim', type=int, default=10, help='')
    parser.add_argument('--rbf_type', type=str, default='gaussian', help='')

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
        env = env_casadi.CSTR()
    else:
        raise NameError('Wrong environment name')

    return env


def get_algo(config, env, device):
    algo_name = config.algo

    config.s_dim = env.s_dim
    config.a_dim = env.a_dim
    config.nT = env.nT
    config.device = device

    if algo_name in ['DQN', 'QRDQN', 'DDPG', 'SAC', 'GDHP']:
        config.update_type = 'single_train_per_single_step'
    elif algo_name in ['A2C', 'TRPO', 'PPO', 'iLQR', 'SDDP']:
        config.update_type = 'single_train_per_single_episode'
    elif algo_name in ['REPS', 'PoWER', 'PI2']:
        config.update_type = 'single_train_per_multiple_episodes'
    else:
        raise NameError('Wrong algorithm name')

    if algo_name == 'A2C':
        algo = a2c.A2C(config)
    elif algo_name == 'DDPG':
        algo = ddpg.DDPG(config)
    elif algo_name == 'DQN':
        algo = dqn.DQN(config)
    elif algo_name == 'GDHP':
        algo = gdhp.GDHP(config)
    elif algo_name == 'iLQR':
        algo = ilqr.iLQR(config)
    elif algo_name == 'PI2':
        algo = pi2.PI2(config)
    elif algo_name == 'PoWER':
        algo = power.PoWER(config)
    elif algo_name == 'PPO':
        algo = ppo.PPO(config)
    elif algo_name == 'QRDQN':
        algo = qrdqn.QRDQN(config)
    elif algo_name == 'REPS':
        algo = reps.REPS(config)
    elif algo_name == 'SAC':
        algo = sac.SAC(config)
    elif algo_name == 'SDDP':
        algo = sddp.SDDP(config)
    elif algo_name == 'TRPO':
        algo = trpo.TRPO
    else:
        raise NameError('Wrong algorithm name')

    return algo

def set_seed(config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
