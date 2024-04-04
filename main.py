import os
import torch

from config2 import get_config, get_env, get_algo, set_seed
from train import Trainer


if __name__ == "__main__":
    # Basic configurations
    config = get_config()

    if not os.path.exists(config.result_save_path + config.env):
        os.makedirs(config.result_save_path + config.env)

    if not os.path.exists(config.model_save_path + config.env):
        os.makedirs(config.model_save_path + config.env)

    # Set seed
    set_seed(config)

    # Environment
    env = get_env(config)

    # Algorithm
    agent = get_algo(config, env)

    # Train
    trainer = Trainer(config, env, agent)
    trainer.train()
    trainer.plot()
