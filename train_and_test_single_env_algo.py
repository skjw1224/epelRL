import os
from config import get_config, get_env, get_algo, set_test_seed
from test import Tester
from train_single_env_algo import train_single_env_algo


def test_single_env_algo():
    # Basic configurations
    config = get_config()
    env_name = config['env']
    algo_name = config['algo']

    # Set save path
    config['save_path'] = os.path.join(os.getcwd(), '_Result', f'{env_name}_{algo_name}')
    os.makedirs(config['save_path'], exist_ok=True)

    # Set seed
    set_test_seed(config)

    # Environment
    env = get_env(config)

    # Algorithm
    agent = get_algo(config, env)

    # Train
    tester = Tester(config, env, agent)
    avg_cost, std_cost = tester.test()
    tester.plot()

    return avg_cost, std_cost


if __name__ == '__main__':
    train_single_env_algo()
    test_single_env_algo()

