import torch
import utils

class Config(object):
    """Save hyperparameters"""
    def __init__(self):
        self.seed = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.environment = None
        self.algorithm = None
        self.max_episode = None
        self.hyperparameters = None

    def encode_settings(self, **args):
        if 'device' in args: self.device = args['deivce']
        self.environment = args['environment']
        self.algorithm = args['algorithm']
        self.max_episode = args['max_episode']
        self.hyperparameters = args['hyperparameters']

        self.alg_default_settings()

    def alg_default_settings(self):
        if self.algorithm['controller'] in ['dqn', 'qrdqn']:
            self.algorithm['action_type'] = 'discrete'
            self.algorithm['action_mesh_idx'] = utils.action_meshgen(self.algorithm['single_dim_mesh'], self.environment.a_dim)

