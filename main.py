import os
from config import get_config, get_env, get_algo, set_seed
from train import Trainer


if __name__ == "__main__":
    # Basic configurations
    config = get_config()
    
    # Set save path
    current_path = os.getcwd()
    save_path = os.path.join(current_path, 'result', f'{config.env}_{config.algo}')
    os.makedirs(save_path, exist_ok=True)
    config.save_path = save_path

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
