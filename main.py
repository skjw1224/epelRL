import numpy as np

import torch
import sys
import os

import time
my_CSTR = os.getcwd()
sys.path.append(my_CSTR)
# from env import CstrEnv
from env_casadi import CstrEnv
from replay_buffer import ReplayBuffer

from config import Config
from train import Train


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
device = 'cpu'

config = Config()
trainer = Train(config)

ilqr_controller = ILQR(env, device)
sddp_controller = SDDP(env, device)
pid_controller = PID(env, device)
dqn_controller = DQN(env, device)
gdhp_controller = GDHP(env, device)
ddpg_controller = DDPG(env, device, replay_buffer)
trpo_controller = TRPO(env, device)
a2c_controller = A2C(env, device)
PoWER_controller = PoWER(env, device)


trainer.env_rollout()


