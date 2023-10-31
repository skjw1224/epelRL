import torch
import os
from env_casadi import CSTR
from config import Config
from train import Train


if __name__ == "__main__":
    config = Config()
    config.environment = CSTR()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        # 'iLQR': None,
        'DQN': None,
        # 'QRDQN': None,
        # 'DDPG': None,
        # 'GDHP': None,
        # 'SDDP': None,
        # 'A2C': None,
        # 'SAC': None,
        # 'TRPO': None,
        # 'PPO': None,
        # 'REPS': None,
        # 'REPS_NN': None,
        # 'PoWER': None,
        # 'PI2': None,
    }

    config.encode_settings(alg_settings)
    trainer = Train(config)
    trainer.env_rollout()
    trainer.plot()
