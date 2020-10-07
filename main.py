import numpy as np
import matplotlib.pyplot as plt
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
from data_postprocessing import DataPostProcessing


from ilqr import ILQR
from sddp import SDDP
from pid import PID
from dqn import DQN
from gdhp import GDHP
from ddpg import DDPG
from a2c import Actor_Critic
from trpo_modif import TRPO
from PoWER import PoWER



MAX_EPISODE = 21
BUFFER_SIZE = 600
MINIBATCH_SIZE = 32

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
device = 'cpu'

config = Config()
trainer = Train(config)
postprocessing = DataPostProcessing(config)

env = CstrEnv(device)
replay_buffer = ReplayBuffer(env, device, buffer_size=BUFFER_SIZE, batch_size=MINIBATCH_SIZE)
ilqr_controller = ILQR(env, device)
sddp_controller = SDDP(env, device)
pid_controller = PID(env, device)
dqn_controller = DQN(env, device)
gdhp_controller = GDHP(env, device)
ddpg_controller = DDPG(env, device, replay_buffer)
trpo_controller = TRPO(env, device)
a2c_controller = Actor_Critic(env, device)
PoWER_controller = PoWER(env, device)

s_dim = env.s_dim
a_dim = env.a_dim
o_dim = env.o_dim

plt_num = 0

controller = ilqr_controller
controller = ddpg_controller
controller = sddp_controller
controller = pid_controller
controller = dqn_controller
controller = gdhp_controller


epi_data = trainer.env_rollout()
postprocessing.stats_record(epi_solutions, epi_data, epi_misc_data, epi_num=i)
postprocessing.print_and_save_history(epi_num=i)
postprocessing.plot(epi_num=i)

