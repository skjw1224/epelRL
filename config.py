import utils

# Controllers
from dqn import DQN
from ddpg import DDPG
from gdhp import GDHP
from ilqr import ILQR
from pid import PID
from sac import SAC
from qrdqn import QRDQN
from sddp import SDDP

# Explorers
from explorers import OU_Noise, E_greedy, Gaussian_noise

# Approximators
from nn_create import NeuralNetworks
from torch_rbf import RBF

class Config(object):
    """Save hyperparameters"""
    def __init__(self):
        self.device = None
        self.environment = None
        self.algorithm = None
        self.hyperparameters = {}
        self.result_save_path = None

        self.alg_key_matching()

    def encode_settings(self, kwargs):
        for key, val in kwargs.items():
            self.algorithm = {'controller':
                                  {'function': self.ctrl_key2arg[key],
                                   'name': key,
                                   'action_type': None,
                                   'action_mesh_idx': None,
                                   'model_requirement': None,
                                   'initial_controller': None,},
                              'explorer':
                                  {'function': None,
                                   'name': None},
                              'approximator':
                                  {'function': None,
                                   'name': None}}

            # Default (algorithm specific) settings
            self.controller_default_settings()
            self.hyper_default_settings()

            # Override alternative settings
            if val is not None:
                # Override explorer function
                if 'explorer' in val:
                    f_name = val['explorer']
                    self.algorithm['explorer']['function'] = self.exp_key2arg[f_name]
                    self.algorithm['explorer']['name'] = f_name

                # Override approximator function
                if 'approximator' in val:
                    f_name = val['approximator']
                    self.algorithm['approximator']['function'] = self.approx_key2arg[f_name]
                    self.algorithm['approximator']['name'] = f_name

                # Override hyperparameters
                for hkey, hval in val.items():
                    self.hyperparameters[hkey] = hval

    def alg_key_matching(self):
        self.ctrl_key2arg = {
            "DQN": DQN,
            "DDPG": DDPG,
            "GDHP": GDHP,
            "ILQR": ILQR,
            "PID": PID,
            "SAC": SAC,
            "QRDQN": QRDQN,
            "SDDP": SDDP
        }

        self.exp_key2arg = {
            'e_greedy': E_greedy,
            'OU': OU_Noise,
            'Gaussian': Gaussian_noise
        }

        self.approx_key2arg = {
            'DNN': NeuralNetworks,
            'RBF': RBF
        }

    def controller_default_settings(self):
        # Discrete or Continuous
        if self.algorithm['controller']['name'] in ['DQN', 'QRDQN']:
            self.algorithm['controller']['action_type'] = 'discrete'

        else:
            self.algorithm['controller']['action_type'] = 'continuous'

        # Model-based or Model-free
        if self.algorithm['controller']['name'] in ['GDHP']:
            self.algorithm['controller']['model_requirement'] = 'model_based'
        else:
            self.algorithm['controller']['model_requirement'] = 'model_free'

        # Default initial controller
        if self.algorithm['controller']['action_type'] == 'continuous':
            if self.algorithm['controller']['model_requirement'] == 'model_based':
                self.algorithm['controller']['initial_controller'] = ILQR
            else:
                self.algorithm['controller']['initial_controller'] = PID

        # Default explorer
        if self.algorithm['controller']['action_type'] == 'continuous':
            self.algorithm['explorer']['name'] = 'OU'
            self.algorithm['explorer']['function'] = self.exp_key2arg['OU']
        else:
            self.algorithm['explorer']['name'] = 'e_greedy'
            self.algorithm['explorer']['function'] = self.exp_key2arg['e_greedy']

        # Default approximator
        self.algorithm['approximator']['name'] = 'DNN'
        self.algorithm['approximator']['function'] = self.approx_key2arg['DNN']


    def hyper_default_settings(self):
        self.hyperparameters['init_ctrl_idx'] = 20
        self.hyperparameters['explore_epi_idx'] = 50
        self.hyperparameters['max_episode'] = 20
        self.hyperparameters['hidden_nodes'] = [50, 50, 30]
        self.hyperparameters['tau'] = 0.05
        self.hyperparameters['buffer_size'] = 600
        self.hyperparameters['minibatch_size'] = 32
        self.hyperparameters['eps_greedy_denom'] = 1
        self.hyperparameters['eps_greedy'] = 0.1
        self.hyperparameters['adam_eps'] = 1E-4
        self.hyperparameters['l2_reg'] = 1E-3
        self.hyperparameters['grad_clip_mag'] = 5.0

        self.hyperparameters['save_period'] = 5
        self.hyperparameters['plot_snapshot'] = [0, 10, 15, 20]

        # Algorithm specific settings
        if self.algorithm['controller']['name'] == 'DQN':
            self.hyperparameters['single_dim_mesh'] = [-1., -.9, -.5, -.2, -.1, -.05, 0., .05, .1, .2, .5, .9, 1.]
            self.algorithm['controller']['action_mesh_idx'] = \
                utils.action_meshgen(self.hyperparameters['single_dim_mesh'], self.environment.a_dim)
            self.hyperparameters['learning_rate'] = 2E-4
        elif self.algorithm['controller']['name'] == 'DDPG':
            self.hyperparameters['critic_learning_rate'] = 1E-2
            self.hyperparameters['actor_learning_rate'] = 1E-3
        elif self.algorithm['controller']['name'] == 'A2C':
            self.hyperparameters['bootstrap_length'] = 10
            self.hyperparameters['critic_learning_rate'] =2E-4
            self.hyperparameters['actor_learning_rate'] = 1E-4
            self.hyperparameters['eps_decay_rate'] = 0.99
            self.algorithm['explorer']['name'] = 'Gaussian'
            self.algorithm['explorer']['function'] = self.exp_key2arg['Gaussian']

        elif self.algorithm['controller']['name'] == 'GDHP':
            self.hyperparameters['critic_learning_rate'] = 2E-4
            self.hyperparameters['actor_learning_rate'] = 2E-4
            self.hyperparameters['costate_learning_rate'] = 2E-4
        elif self.algorithm['controller']['name'] == 'ILQR':
            self.hyperparameters['learning_rate'] = 0.1

        if self.algorithm['controller']['initial_controller'] == ILQR:
            self.hyperparameters['learning_rate'] = 0.1

        if self.algorithm['explorer']['name'] == 'OU':
            self.hyperparameters['ou_mu0'] = 0.
            self.hyperparameters['ou_theta'] = 0.15
            self.hyperparameters['ou_sigma'] = 0.2
