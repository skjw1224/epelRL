import argparse
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

import algorithm
import environment

def get_algo_specific_default(args):
    params_name = ['critic_lr', 'adam_eps', 'l2_reg', 'rbf_dim', 'num_rollout']
    params_default = {
        'A2C': (0.01, 1.e-6, 1.e-3, [], []),
        'DDPG': (0.001, 0.0001, 1.e-3, [], []),
        'DQN': (0.01, 1.e-6, 1.e-3, [], []),
        'iLQR': (1.e-4, 1.e-6, 1.e-3, [], []),
        'GDHP': (1.e-4, 1.e-6, 1.e-3, [], []),
        'SDDP': (1e-4, 1e-6, 1e-3, [], []),
        'QRDQN': (0.0001, 1.e-6, 0.1, [], []),
        'SAC': (1e-4, 1e-6, 1e-3, [], []),
        'TD3': (0.01, 1.e-6, 1.e-3, [], []),
        'PPO': (0.001, 1.e-6, 1.e-3, [], []),
        'TRPO': (1e-4, 1e-6, 1e-3, [], []),
        'PoWER': (1.e-4, 1.e-6, 1.e-3, 1000, 1000),
        'REPS': (1.e-4, 1.e-6, 1.e-3, 50, 50),
        'PI2': (1.e-4, 1.e-6, 1.e-3, 50, 10)
    }

    for idx, name in enumerate(params_name):
        val = getattr(args, name)
        if val is None:
            val_default = params_default[args.algo][idx]
            setattr(args, name, val_default)

    return args


def get_config():
    parser = argparse.ArgumentParser(description='EPEL RL')

    # Basic settings
    parser.add_argument('--algo', type=str, default='TD3', help='RL algorithm')
    parser.add_argument('--env', type=str, default='CSTR', help='Environment')
    parser.add_argument('--seed', type=int, default=0, help='Seed number')
    parser.add_argument('--device', type=str, default='cuda', help='Device - cuda or cpu')
    parser.add_argument('--save_freq', type=int, default=20, help='Save frequency')
    parser.add_argument('--save_model', action='store_true', help='Whether to save model or not')
    parser.add_argument('--load_model', action='store_true', help='Whether to load saved model or not')
    parser.add_argument('--show_plot', type=bool, default=False, help='Whether to show plot')
    parser.add_argument('--disp_opt', type=bool, default=True, help='Whether to print training stats')

    # Training settings
    parser.add_argument('--max_episode', type=int, default=20, help='Maximum training episodes')
    parser.add_argument('--init_ctrl_idx', type=int, default=0, help='Episodes for training with initial controller')
    parser.add_argument('--buffer_size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=1024, help='Mini-batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount')
    parser.add_argument('--warm_up_episode', type=int, default=10, help='Number of warm up episode')
    parser.add_argument('--num_evaluate', type=int, default=3, help='Number of evaluation per episode')
    parser.add_argument('--convg_bound', type=float, default=5.e-2, help='Upper bound of convergence criteria')

    # Neural network parameters
    parser.add_argument('--num_hidden_nodes', type=int, default=128, help='Number of hidden nodes in MLP')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers in MLP')
    parser.add_argument('--tau', type=float, default=0.005, help='Parameter for soft target update')
    parser.add_argument('--grad_clip_mag', type=float, default=5.0, help='Gradient clipping magnitude')
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='Actor network learning rate')
    # -- Algorithm specific parameters
    parser.add_argument('--adam_eps', type=float, help='Epsilon for numerical stability')
    parser.add_argument('--critic_lr', type=float, help='Critic network learning rate')
    parser.add_argument('--l2_reg', type=float, help='Weight decay (L2 penalty)')

    # RBF parameters
    parser.add_argument('--rbf_type', type=str, default='gaussian', help='Type of RBF basis function')
    # -- Algorithm specific parameters
    parser.add_argument('--rbf_dim', type=int, help='Dimension of RBF basis function')
    parser.add_argument('--num_rollout', type=int, help='Number of episodes rolled out for rbf regression')

    # Test setting
    parser.add_argument('--test_seed', type=int, default=3, help='Seed number in test mode')
    parser.add_argument('--num_test_evaluate', type=int, default=5, help='Number of evaluation in test mode')

    args = parser.parse_args()
    args = get_algo_specific_default(args)

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
        args.h = 10
        args.init_lambda = 25
    elif args.algo == 'PoWER':
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
        args.critic_reg = 0.01
        args.actor_reg = 1
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
        args.num_cg_iterations = 100
        args.num_line_search = 10
        args.max_kl_divergence = 0.001

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
       
    return vars(args)


def get_env(config):
    env_name = config['env']

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
        env = environment.PFR(config)
    else:
        raise NameError('Wrong environment name')

    return env


def get_algo(config, env):
    algo_name = config['algo']
    config['s_dim'] = env.s_dim
    config['a_dim'] = env.a_dim
    config['p_dim'] = env.p_dim
    config['nT'] = env.nT
    config['dt'] = env.dt

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
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

def set_test_seed(config):
    torch.manual_seed(config['test_seed'])
    np.random.seed(config['test_seed'])
    random.seed(config['test_seed'])

def plot_traj_data(env, traj_data_history, plot_case, case_name, save_name, show_plot=False):
    """traj_data_history: (num_evaluate, NUM_CASE, nT, traj_dim)"""
    color_cycle_tab20 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
                         '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
                         '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
                         '#17becf', '#9edae5']

    variable_tag_lst = env.plot_info['variable_tag_lst']
    state_plot_idx_lst = env.plot_info['state_plot_idx_lst'] if 'state_plot_idx_lst' in env.plot_info else range(1, env.s_dim)
    ref_idx_lst = env.plot_info['ref_idx_lst']
    nrows_s, ncols_s = env.plot_info['state_plot_shape']
    nrows_a, ncols_a = env.plot_info['action_plot_shape']

    ref = env.ref_traj()
    x_axis = np.linspace(env.t0+env.dt, env.tT, num=env.nT)

    traj_mean = traj_data_history.mean(axis=0)
    traj_std = traj_data_history.std(axis=0)

    # State variables subplots
    fig1, ax1 = plt.subplots(nrows_s, ncols_s, figsize=(ncols_s*6, nrows_s*5))
    for i, fig_idx in enumerate(ref_idx_lst):
        ax1.flat[fig_idx-1].hlines(ref[i], env.t0, env.tT, color='r', linestyle='--', label='Set point')

    for fig_idx, i in enumerate(state_plot_idx_lst):
        ax1.flat[fig_idx].set_xlabel(variable_tag_lst[0])
        ax1.flat[fig_idx].set_ylabel(variable_tag_lst[fig_idx+1])
        if len(plot_case) > 12:
            ax1.flat[fig_idx].set_prop_cycle(color=color_cycle_tab20)
        for case in plot_case:
            ax1.flat[fig_idx].plot(x_axis, traj_mean[case, :, i], label=case_name[case])
            ax1.flat[fig_idx].fill_between(x_axis, traj_mean[case, :, i] + traj_std[case, :, i], traj_mean[case, :, i] - traj_std[case, :, i], alpha=0.5)
        ax1.flat[fig_idx].legend()
        ax1.flat[fig_idx].grid()
    fig1.tight_layout()
    plt.savefig(save_name + '_state_traj.png')
    if show_plot:
        plt.show()
    plt.close()

    # Action variables subplots
    x_axis = np.linspace(env.t0, env.tT, num=env.nT)
    fig3, ax3 = plt.subplots(nrows_a, ncols_a, figsize=(ncols_a*6, nrows_a*5))
    for i in range(env.a_dim):
        axis = ax3.flat[i] if env.a_dim > 1 else ax3
        axis.set_xlabel(variable_tag_lst[0])
        axis.set_ylabel(variable_tag_lst[len(state_plot_idx_lst) + 1])
        if len(plot_case) > 12:
            axis.set_prop_cycle(color=color_cycle_tab20)
        for case in plot_case:
            axis.plot(x_axis, traj_mean[case, :, env.s_dim + i], label=case_name[case])
            axis.fill_between(x_axis, traj_mean[case, :, env.s_dim + i] + traj_std[case, :, env.s_dim + i], traj_mean[case, :, env.s_dim + i] - traj_std[case, :, env.s_dim + i], alpha=0.5)
        axis.legend()
        axis.grid()
    fig3.tight_layout()
    plt.savefig(save_name + '_action_traj.png')
    if show_plot:
        plt.show()
    plt.close()
