import torch
import numpy as np
from collections import deque
import random

class trajectory_buffer(object):
    def __init__(self, env, device, number):
        self.Trajectory = [[]]
        for k in range(number - 1):
            self.Trajectory.append([])

    def add(self, number, *args):
        state, action, reward = args
        self.Trajectory[number].append(state)
        self.Trajectory[number].append(action)
        self.Trajectory[number].append(reward)

    def get_trajectory(self, number):
        return self.Trajectory[number]

    def reset_trajectory(self, number):
        self.Trajectory = [[]]
        for k in range(number - 1):
            self.Trajectory.append([])
