import torch
import utils

class Config(object):
    """Save hyperparameters"""
    def __init__(self):
        self.seed = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.environment = None
        self.algorithm = None
        self.hyperparameters = None
        # should contain: 'explore_epi_idx', 'tau', 'max_episode'

    def encode_settings(self, **args):
        if 'device' in args: self.device = args['deivce']
        self.environment = args['environment']
        self.algorithm = args['algorithm']

        self.hyp_default_settings()
        # Override alternative values
        self.hyperparameters = args['hyperparameters']

        # Set default algorithm specific settings
        self.alg_default_settings()

    def alg_default_settings(self):
        if self.algorithm['controller'] in ['dqn', 'qrdqn']:
            self.algorithm['action_type'] = 'discrete'
            self.algorithm['action_mesh_idx'] = utils.action_meshgen(self.algorithm['single_dim_mesh'], self.environment.a_dim)

    def hyp_default_settings(self):
        self.hyperparameters['init_ctrl_idx'] = 20
        self.hyperparameters['explore_epi_idx'] = 50
        self.hyperparameters['max_episode'] = 200
        self.hyperparameters['tau'] = 0.05
        self.hyperparameters['buffer_size'] = 600
        self.hyperparameters['minibatch_size'] = 32
        self.hyperparameters['eps_greedy_denom'] = 1
        self.hyperparameters['eps_greedy'] = 0.1
        self.hyperparameters['adam_eps'] = 1E-4
        self.hyperparameters['l2_reg'] = 1E-3
        self.hyperparameters['grad_clip_mag'] = 5.0

        # Algorithm specific settings
        if self.algorithm['controller'] == 'dqn':
            self.hyperparameters['learning_rate'] = 2E-4
        elif self.algorithm['controller'] == 'ddpg':
            self.hyperparameters['critic_learning_rate'] = 1E-2
            self.hyperparameters['actor_learning_rate'] = 1E-3



