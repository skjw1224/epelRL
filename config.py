import utils

from dqn import DQN
from ddpg import DDPG
from gdhp import GDHP
from ilqr import ILQR
from pid import PID
from sac import SAC
from qrdqn import QRDQN
from sddp import SDDP

class Config(object):
    """Save hyperparameters"""
    def __init__(self):
        self.device = None
        self.environment = None
        self.algorithm = None
        self.hyperparameters = None

        self.alg_key_matching()

    def encode_settings(self, **kwargs):
        for key, val in kwargs.items():
            self.algorithm = self.key2arg[key]

            self.hyper_default_settings()
            # Override alternative values
            if val is not None:
                for hkey, hval in val:
                    self.hyperparameters[hkey] = hval

            # Set default algorithm specific settings
            self.alg_specific_settings()

    def alg_key_matching(self):
        self.key2arg = {
            "DQN": DQN,
            "DDPG": DDPG,
            "GDHP": GDHP,
            "ILQR": ILQR,
            "PID": PID,
            "SAC": SAC,
            "QRDQN": QRDQN,
            "SDDP": SDDP
        }

    def alg_specific_settings(self):
        if self.algorithm['controller'] in ['dqn', 'qrdqn']:
            self.algorithm['action_type'] = 'discrete'
            self.algorithm['action_mesh_idx'] = utils.action_meshgen(self.algorithm['single_dim_mesh'], self.environment.a_dim)
        else:
            self.algorithm['action_type'] = 'continuous'

        if self.algorithm['controller'] in ['gdhp']:
            self.algorithm['model_requirement'] = 'model_based'
        else:
            self.algorithm['model_requirement'] = 'model_free'

    def hyper_default_settings(self):
        self.hyperparameters['init_ctrl_idx'] = 20
        self.hyperparameters['explore_epi_idx'] = 50
        self.hyperparameters['max_episode'] = 200
        self.hyperparameters['hidden_nodes'] = [50, 50, 30]
        self.hyperparameters['tau'] = 0.05
        self.hyperparameters['buffer_size'] = 600
        self.hyperparameters['minibatch_size'] = 32
        self.hyperparameters['eps_greedy_denom'] = 1
        self.hyperparameters['eps_greedy'] = 0.1
        self.hyperparameters['eps_decay_rate'] = 0.99
        self.hyperparameters['adam_eps'] = 1E-4
        self.hyperparameters['l2_reg'] = 1E-3
        self.hyperparameters['grad_clip_mag'] = 5.0

        self.hyperparameters['save_period'] = 5

        # Algorithm specific settings
        if self.algorithm['controller'] == 'dqn':
            self.hyperparameters['learning_rate'] = 2E-4
        elif self.algorithm['controller'] == 'ddpg':
            self.hyperparameters['critic_learning_rate'] = 1E-2
            self.hyperparameters['actor_learning_rate'] = 1E-3
        elif self.algorithm['controller'] == 'a2c':
            self.hyperparameters['bootstrap_length'] = 10
            self.hyperparameters['critic_learning_rate'] =2E-4
            self.hyperparameters['actor_learning_rate'] = 1E-4
        elif self.algorithm['controller'] == 'gdhp':
            self.hyperparameters['critic_learning_rate'] = 2E-4
            self.hyperparameters['actor_learning_rate'] = 2E-4
            self.hyperparameters['costate_learning_rate'] = 2E-4
