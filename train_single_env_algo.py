import os
import time
from config import get_config, get_env, get_algo, set_seed
from train import Trainer


def train_single_env_algo():
    # Basic configurations
    start_time = time.time()
    config = get_config()
    env_name = config['env']
    algo_name = config['algo']

    # Set save path
    config['save_path'] = os.path.join(os.getcwd(), '_Result', f'{env_name}_{algo_name}_lr{config["critic_lr"]}_AdamEps{config["adam_eps"]}_l2ref{config["l2_reg"]}')
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

    return minimum_cost


if __name__ == '__main__':
    train_single_env_algo()
