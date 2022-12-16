import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a deep Q-learning network
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DeepNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        n_h_nodes = [50, 50, 30]

        self.fc0 = nn.Linear(self.input_dim, n_h_nodes[0])
        self.bn0 = nn.BatchNorm1d(n_h_nodes[0])
        self.fc1 = nn.Linear(n_h_nodes[0], n_h_nodes[1])
        self.bn1 = nn.BatchNorm1d(n_h_nodes[1])
        self.fc2 = nn.Linear(n_h_nodes[1], n_h_nodes[2])
        self.bn2 = nn.BatchNorm1d(n_h_nodes[2])
        self.fc3 = nn.Linear(n_h_nodes[2], self.output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x
