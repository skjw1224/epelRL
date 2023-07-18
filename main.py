import numpy as np

import torch
import sys
import os
from env_casadi import CstrEnv


from config import Config
from train import Train

config = Config()

config.environment = CstrEnv()
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# config.device = 'cpu'
path = 'results/' + config.environment.env_name
try:
    os.mkdir('results')
    os.mkdir(path)
except FileExistsError:
    pass
config.result_save_path = path + '/'

config.standard_deviation_results = 1.0
config.save_model = False


alg_settings = {
    # "iLQR": None,
    # "DQN": None,
    # "QRDQN": None,
    # 'DDPG': None
    # 'GDHP': None,
    # 'GPS': None,
    # 'SDDP': None,
    # "A2C": None,
    # "SAC": None,
    # 'TRPO': None,
    # 'PPO': None,
    # 'REPS': None,
    # 'REPS_NN': None,
    # 'PoWER': None,


    # "Stochastic_Policy_Search_Agents": {
    #     "policy_network_type": "Linear",
    #     "noise_scale_start": 1e-2,
    #     "noise_scale_min": 1e-3,
    #     "noise_scale_max": 2.0,
    #     "noise_scale_growth_factor": 2.0,
    #     "stochastic_action_decision": False,
    #     "num_policies": 10,
    #     "episodes_per_policy": 1,
    #     "num_policies_to_keep": 5,
    #     "clip_rewards": False
    # },
    # "Policy_Gradient_Agents": {
    #     "learning_rate": 0.05,
    #     "linear_hidden_units": [20, 20],
    #     "final_layer_activation": "SOFTMAX",
    #     "learning_iterations_per_round": 5,
    #     "discount_rate": 0.99,
    #     "batch_norm": False,
    #     "clip_epsilon": 0.1,
    #     "episodes_per_learning_round": 4,
    #     "normalise_rewards": True,
    #     "gradient_clipping_norm": 7.0,
    #     "mu": 0.0, #only required for continuous action games
    #     "theta": 0.0, #only required for continuous action games
    #     "sigma": 0.0, #only required for continuous action games
    #     "epsilon_decay_rate_denominator": 1.0,
    #     "clip_rewards": False
    # },
    #
    # "Actor_Critic_Agents":  {
    #
    #     "learning_rate": 0.005,
    #     "linear_hidden_units": [20, 10],
    #     "final_layer_activation": ["SOFTMAX", None],
    #     "gradient_clipping_norm": 5.0,
    #     "discount_rate": 0.99,
    #     "epsilon_decay_rate_denominator": 1.0,
    #     "normalise_rewards": True,
    #     "exploration_worker_difference": 2.0,
    #     "clip_rewards": False,
    #
    #     "Actor": {
    #         "learning_rate": 0.0003,
    #         "linear_hidden_units": [64, 64],
    #         "final_layer_activation": "Softmax",
    #         "batch_norm": False,
    #         "tau": 0.005,
    #         "gradient_clipping_norm": 5,
    #         "initialiser": "Xavier"
    #     },
    #
    #     "Critic": {
    #         "learning_rate": 0.0003,
    #         "linear_hidden_units": [64, 64],
    #         "final_layer_activation": None,
    #         "batch_norm": False,
    #         "buffer_size": 1000000,
    #         "tau": 0.005,
    #         "gradient_clipping_norm": 5,
    #         "initialiser": "Xavier"
    #     },
    #
    #     "min_steps_before_learning": 400,
    #     "batch_size": 256,
    #     "discount_rate": 0.99,
    #     "mu": 0.0, #for O-H noise
    #     "theta": 0.15, #for O-H noise
    #     "sigma": 0.25, #for O-H noise
    #     "action_noise_std": 0.2,  # for TD3
    #     "action_noise_clipping_range": 0.5,  # for TD3
    #     "update_every_n_steps": 1,
    #     "learning_updates_per_learning_session": 1,
    #     "automatically_tune_entropy_hyperparameter": True,
    #     "entropy_term_weight": None,
    #     "add_extra_noise": False,
    #     "do_evaluation_iterations": True
    # }
}

config.encode_settings(alg_settings)
trainer = Train(config)
trainer.env_rollout()
# trainer.plot()

