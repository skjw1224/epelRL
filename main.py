import os
import torch

from config2 import get_config, get_env, get_algo
from train import Train


if __name__ == "__main__":
    config = get_config()
    env = get_env(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    controller = get_algo(config, env, device)
    trainer = Train(config, env, controller)
    trainer.env_rollout()
    trainer.plot()
