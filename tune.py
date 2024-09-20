import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import ray
from ray import train, tune
from ray.train.torch import TorchTrainer
from ray.tune import Tuner
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch

import environment
import algorithm
from config import get_config, get_env, get_algo, set_seed
from train import Trainer


def train_func(config):
    algo_name = config['algo']
    # available_envs = [env.__name__ for env in environment.__all__]
    available_envs = ['CSTR']
    total_cost = 0.

    for env_name in available_envs:
        # Set save path
        config['save_path'] = os.path.join(os.getcwd(), 'result', f'{env_name}_{algo_name}')
        os.makedirs(config['save_path'], exist_ok=True)

        # Set seed
        set_seed(config)

        # Set environment and agent
        config['env'] = env_name
        env = get_env(config)
        agent = get_algo(config, env)

        # Train agent
        trainer = Trainer(config, env, agent)
        trainer.train()
    
        # Result
        minimum_cost = trainer.get_train_results()
        total_cost += minimum_cost
    
    train.report({"cost": total_cost})


if __name__ == '__main__':
    config = get_config()

    # Set Trainer
    torch_trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=train.ScalingConfig(
            num_workers=1,
            use_gpu=True,
            resources_per_worker={'CPU': 1, 'GPU': 0.25}
        )
    )
    
    # Set Tuner
    tuner = Tuner(
        torch_trainer,
        param_space={
            "train_loop_config": {
                "num_hidden_nodes": tune.sample_from(lambda _: 2**np.random.randint(6, 9)),
                "num_hidden_layers": tune.grid_search([1, 2, 3]),
                "critic_lr": tune.grid_search([3e-3, 1e-3, 3e-4, 1e-4])
            }
        },
        tune_config=tune.TuneConfig(
            metric="cost",
            mode="min",
            # search_alg=BayesOptSearch(),
            num_samples=8,
            max_concurrent_trials=8,
        ),
    )

    # Tune hyper-parameters
    results = tuner.fit()
    
    # Print results
    print("Best result")
    cost = results.get_best_result().metrics['cost']
    print(f"-- Cost: {cost}")
    for _key, _val in results.get_best_result().config['train_loop_config'].items():
        print(f'-- {_key}: {_val}')
    
    print('DONE')
    