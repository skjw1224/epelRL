import torch
import numpy as np
from collections import deque
from itertools import islice
import random

class ReplayBuffer(object):
    def __init__(self, env, device, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.env = env
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.device = device

    def add(self, *args):
        # If numpy replay buffer
        # experience = [arg.detach().numpy() for arg in args]
        # experience = [arg.detach() for arg in args]
        experience = list(args)
        self.memory.append(experience)

    def clear(self):
        self.memory.clear()

    def sample(self):
        """Random sample with batch size: shuffle indices. For off-policy methods"""
        batch = random.sample(self.memory, k=self.batch_size)

        # numpy replay buffer
        # t_batch = torch.from_numpy(np.vstack([e[0] for e in batch])).float().to(self.device)

        # Pytorch replay buffer - squeeze 2nd dim (B, 1, x) -> (B, x)
        x_batch = torch.stack([e[0] for e in batch]).squeeze(1)
        u_batch = torch.stack([e[1] for e in batch]).squeeze(1)
        r_batch = torch.stack([e[2] for e in batch]).squeeze(1)
        x2_batch = torch.stack([e[3] for e in batch]).squeeze(1)
        term_batch = torch.stack([e[4] for e in batch]).squeeze(1)

        if len(batch[0]) <= 5:
            return x_batch, u_batch, r_batch, x2_batch, term_batch
        else: # For model-based RL
            dfdx_batch = torch.stack([e[5] for e in batch if e is not None])
            dfdu_batch = torch.stack([e[6] for e in batch if e is not None])
            dcdx_batch = torch.stack([e[7] for e in batch if e is not None])
            d2cdu2inv_batch = torch.stack([e[8] for e in batch if e is not None])
            return x_batch, u_batch, r_batch, x2_batch, term_batch, dfdx_batch, dfdu_batch, dcdx_batch, d2cdu2inv_batch

    def sample_sequence(self):
        """Ordered sequence replay with batch size: Do not shuffle indices. For on-policy methods"""

        min_start = len(self.memory) - self.batch_size  # If batch_size = episode length
        if min_start == 0: min_start = 1
        start_idx = np.random.randint(0, min_start)

        batch = deque(islice(self.memory, start_idx, start_idx + self.batch_size))

        # Pytorch replay buffer - squeeze 2nd dim (B, 1, x) -> (B, x)
        x_batch = torch.stack([e[0] for e in batch]).squeeze(1)
        u_batch = torch.stack([e[1] for e in batch]).squeeze(1)
        r_batch = torch.stack([e[2] for e in batch]).squeeze(1)
        x2_batch = torch.stack([e[3] for e in batch]).squeeze(1)
        term_batch = torch.stack([e[4] for e in batch]).squeeze(1)

        if len(batch[0]) <= 5:
            return x_batch, u_batch, r_batch, x2_batch, term_batch
        else: # For model-based RL
            dfdx_batch = torch.stack([e[5] for e in batch if e is not None])
            dfdu_batch = torch.stack([e[6] for e in batch if e is not None])
            dcdx_batch = torch.stack([e[7] for e in batch if e is not None])
            d2cdu2inv_batch = torch.stack([e[8] for e in batch if e is not None])
            return x_batch, u_batch, r_batch, x2_batch, term_batch, dfdx_batch, dfdu_batch, dcdx_batch, d2cdu2inv_batch


    def __len__(self):
        return len(self.memory)