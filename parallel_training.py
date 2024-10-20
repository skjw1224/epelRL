import os
import tempfile
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import ray
import subprocess

import algorithm
import environment
from config import get_config, get_env, get_algo, set_seed
from train import Trainer

@ray.remote(num_gpus=0.33)
def train_single_env_algo(env_name, algo_name):
    print(f"Running: {env_name}-{algo_name}")
    subprocess.run(['python', 'train_single_env_algo.py', '--algo', algo_name, '--env', env_name,
                    '--disp_opt', '0'])

if __name__ == "__main__":
    available_algos = ['A2C', 'DDPG', 'DQN', 'QRDQN', 'SAC', 'TD3', 'TRPO']
    available_envs = ['CSTR', 'CRYSTAL']

    ray.init(runtime_env={"py_modules": [algorithm, environment]}, num_gpus=1)
    works = []
    for env in available_envs:
        for algo in available_algos:
            works.append(train_single_env_algo.remote(env, algo))
    ray.get(works)
    
    print("DONE")
