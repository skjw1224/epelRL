import torch
import numpy as np
from collections import deque
from itertools import islice
import random


class ReplayBuffer(object):
    def __init__(self, config):
        self.memory = deque(maxlen=config.buffer_size)
        self.batch_size = config.batch_size
        self.s_dim = config.s_dim
        self.a_dim = config.a_dim
        self.device = config.device

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
        batch = random.sample(self.memory, k=min(len(self.memory), self.batch_size))

        # Pytorch replay buffer - squeeze 3rd dim (B, x, 1) -> (B, x)
        x_batch = torch.from_numpy(np.array([e[0] for e in batch]).squeeze(-1)).float().to(self.device)
        u_batch = torch.from_numpy(np.array([e[1] for e in batch]).squeeze(-1)).float().to(self.device)
        r_batch = torch.from_numpy(np.array([e[2] for e in batch]).squeeze(-1)).float().to(self.device)
        x2_batch = torch.from_numpy(np.array([e[3] for e in batch]).squeeze(-1)).float().to(self.device)
        term_batch = torch.from_numpy(np.expand_dims(np.array([e[4] for e in batch]), axis=1)).float().to(self.device)

        if len(batch[0]) <= 5:
            return x_batch, u_batch, r_batch, x2_batch, term_batch
        else: # For model-based RL
            dfdx_batch = torch.from_numpy(np.array([e[5] for e in batch if e is not None])).float().to(self.device)
            dfdu_batch = torch.from_numpy(np.array([e[6] for e in batch if e is not None])).float().to(self.device)
            dcdx_batch = torch.from_numpy(np.array([e[7] for e in batch if e is not None])).float().to(self.device)
            dcdu_batch = torch.from_numpy(np.array([e[8] for e in batch if e is not None])).float().to(self.device)
            d2cdx2_batch = torch.from_numpy(np.array([e[9] for e in batch if e is not None])).float().to(self.device)
            d2cdxdu_batch = torch.from_numpy(np.array([e[10] for e in batch if e is not None])).float().to(self.device)
            d2cdu2_batch = torch.from_numpy(np.array([e[11] for e in batch if e is not None])).float().to(self.device)
            d2cdu2inv_batch = torch.from_numpy(np.array([e[12] for e in batch if e is not None])).float().to(self.device)
            return x_batch, u_batch, r_batch, x2_batch, term_batch, dfdx_batch, dfdu_batch, dcdx_batch, dcdu_batch, d2cdx2_batch, d2cdxdu_batch, d2cdu2_batch, d2cdu2inv_batch

    def sample_sequence(self):
        """Ordered sequence replay with batch size: Do not shuffle indices. For on-policy methods"""

        min_start = max(len(self.memory) - self.batch_size, 1)  # If batch_size = episode length
        start_idx = np.random.randint(0, min_start)

        batch = deque(islice(self.memory, start_idx, start_idx + self.batch_size))

        # Pytorch replay buffer - squeeze 3rd dim (B, x, 1) -> (B, x)
        x_batch = torch.from_numpy(np.array([e[0] for e in batch]).squeeze(-1)).float().to(self.device)
        u_batch = torch.from_numpy(np.array([e[1] for e in batch]).squeeze(-1)).float().to(self.device)
        r_batch = torch.from_numpy(np.array([e[2] for e in batch]).squeeze(-1)).float().to(self.device)
        x2_batch = torch.from_numpy(np.array([e[3] for e in batch]).squeeze(-1)).float().to(self.device)
        term_batch = torch.from_numpy(np.expand_dims(np.array([e[4] for e in batch]), axis=1)).float().to(self.device)

        if len(batch[0]) <= 5:
            return x_batch, u_batch, r_batch, x2_batch, term_batch
        else: # For model-based RL
            dfdx_batch = torch.from_numpy(np.array([e[5] for e in batch if e is not None])).float().to(self.device)
            dfdu_batch = torch.from_numpy(np.array([e[6] for e in batch if e is not None])).float().to(self.device)
            dcdx_batch = torch.from_numpy(np.array([e[7] for e in batch if e is not None])).float().to(self.device)
            dcdu_batch = torch.from_numpy(np.array([e[8] for e in batch if e is not None])).float().to(self.device)
            d2cdx2_batch = torch.from_numpy(np.array([e[9] for e in batch if e is not None])).float().to(self.device)
            d2cdxdu_batch = torch.from_numpy(np.array([e[10] for e in batch if e is not None])).float().to(self.device)
            d2cdu2_batch = torch.from_numpy(np.array([e[11] for e in batch if e is not None])).float().to(self.device)
            d2cdu2inv_batch = torch.from_numpy(np.array([e[12] for e in batch if e is not None])).float().to(self.device)
            return x_batch, u_batch, r_batch, x2_batch, term_batch, dfdx_batch, dfdu_batch, dcdx_batch, dcdu_batch, d2cdx2_batch, d2cdxdu_batch, d2cdu2_batch, d2cdu2inv_batch

    def sample_numpy_sequence(self):
        """Ordered sequence replay with batch size: Do not shuffle indices. For methods using numpy"""

        min_start = max(len(self.memory) - self.batch_size, 1)  # If batch_size = episode length
        start_idx = np.random.randint(0, min_start)

        batch = deque(islice(self.memory, start_idx, start_idx + self.batch_size))

        # Numpy replay buffer - squeeze 3rd dim (B, x, 1) -> (B, x)
        x_batch = np.array([e[0] for e in batch]).squeeze(-1)
        u_batch = np.array([e[1] for e in batch]).squeeze(-1)
        r_batch = np.array([e[2] for e in batch]).squeeze(-1)
        x2_batch = np.array([e[3] for e in batch]).squeeze(-1)
        term_batch = np.expand_dims(np.array([e[4] for e in batch]), axis=1)

        if len(batch[0]) <= 5:
            return x_batch, u_batch, r_batch, x2_batch, term_batch
        else: # For model-based RL
            dfdx_batch = np.array([e[5] for e in batch if e is not None])
            dfdu_batch = np.array([e[6] for e in batch if e is not None])
            dcdx_batch = np.array([e[7] for e in batch if e is not None])
            dcdu_batch = np.array([e[8] for e in batch if e is not None])
            d2cdx2_batch = np.array([e[9] for e in batch if e is not None])
            d2cdxdu_batch = np.array([e[10] for e in batch if e is not None])
            d2cdu2_batch = np.array([e[11] for e in batch if e is not None])
            d2cdu2inv_batch = np.array([e[12] for e in batch if e is not None])
            Fc_batch = np.array([e[13] for e in batch if e is not None])
            dFcdx_batch = np.array([e[14] for e in batch if e is not None])
            dFcdu_batch = np.array([e[15] for e in batch if e is not None])
            return x_batch, u_batch, r_batch, x2_batch, term_batch, \
                    dfdx_batch, dfdu_batch, dcdx_batch, dcdu_batch, d2cdx2_batch, d2cdxdu_batch, d2cdu2_batch, d2cdu2inv_batch, \
                    Fc_batch, dFcdx_batch, dFcdu_batch

    def __len__(self):
        return len(self.memory)