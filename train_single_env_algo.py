import os
from config import get_config, get_env, get_algo, set_seed
from train import Trainer


def train_single_env_algo():
    # Basic configurations
    config = get_config()
    env_name = config['env']
    algo_name = config['algo']

    # Set save path
    config['save_path'] = os.path.join(os.getcwd(), '_Result', f'{env_name}_{algo_name}')
    os.makedirs(config['save_path'], exist_ok=True)

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
    minimum_cost = trainer.get_train_results()
    trainer.save_model()

    return minimum_cost


if __name__ == '__main__':
    train_single_env_algo()
