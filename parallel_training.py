import os
import tempfile
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import ray


import algorithm
import environment
from config import get_config, get_env, get_algo, set_seed
from train import Trainer

@ray.remote(num_gpus=0.33)
def train_single_env_algo(config):
    start_time = time.time()

    # Basic configurations
    env_name = config['env']
    algo_name = config['algo']

    # Set save path
    config['save_path'] = os.path.join(os.getcwd(), 'result', f'{env_name}_{algo_name}')
    os.makedirs(config['save_path'], exist_ok=True)

    # Set seed
    set_seed(config)

    # Environment
    env = get_env(config)

    # Algorithm
    agent = get_algo(config, env)

    # Train
    trainer = Trainer(config, env, agent)
    trainer.train(start_time)
    trainer.plot()
    minimum_cost = trainer.get_train_results()
    trainer.save_model()

    print('--------------------')
    print(f"{config['env']} - {config['algo']}")
    print(minimum_cost)
    print('--------------------')
    

if __name__ == "__main__":
    available_algos = ['A2C', 'DDPG', 'DQN', 'QRDQN', 'SAC', 'TD3', 'TRPO']
    available_envs = ['CSTR', 'CRYSTAL']

    ray.init(runtime_env={"py_modules": [algorithm, environment]}, num_gpus=2)
    works = []
    for env in available_envs:
        for algo in available_algos:
            config = get_config()
            config['env'] = env
            config['algo'] = algo
            config['disp_opt'] = False
            works.append(train_single_env_algo.remote(config))
    ray.get(works)
    
    print("DONE")
