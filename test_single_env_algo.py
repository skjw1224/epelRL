import os
from config import get_config, get_env, get_algo, set_test_seed
from test import Tester


def test_single_env_algo():
    # Basic configurations
    config = get_config()
    env_name = config['env']
    algo_name = config['algo']
    print(f'Test in {env_name} by {algo_name}')

    # Set save path
    config['load_path'] = os.path.join(os.getcwd(), '_Result', f'{env_name}_{algo_name}')
    config['save_path'] = os.path.join(os.getcwd(), '_Result', f'test_{env_name}_{algo_name}')
    os.makedirs(config['save_path'], exist_ok=True)

    # Set seed
    set_test_seed(config)

    # Environment
    env = get_env(config)

    # Algorithm
    agent = get_algo(config, env)

    # Test
    tester = Tester(config, env, agent)
    test_traj = tester.test()
    tester.plot()

    return test_traj


if __name__ == '__main__':
    test_traj = test_single_env_algo()

