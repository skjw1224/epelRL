import torch
import torch.nn as nn
from torch.distributions import Normal


class CriticMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_lst, activation_function):
        super(CriticMLP, self).__init__()

        self.fc_lst = []
        current_input_dim = input_dim
        for next_input_dim in hidden_dim_lst:
            fc = nn.Linear(current_input_dim, next_input_dim)
            current_input_dim = next_input_dim
            self.fc_lst.append(fc)
        self.fc_lst = nn.ModuleList(self.fc_lst)

        self.last_fc = nn.Linear(current_input_dim, output_dim)
        self.activation_function = activation_function

    def forward(self, inputs):
        h = inputs
        for fc in self.fc_lst:
            h = self.activation_function(fc(h))
        outputs = self.last_fc(h)

        return outputs


class ActorMlp(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_lst, activation_function):
        super(ActorMlp, self).__init__()

        self.fc_lst = []
        current_input_dim = input_dim
        for next_input_dim in hidden_dim_lst:
            fc = nn.Linear(current_input_dim, next_input_dim)
            current_input_dim = next_input_dim
            self.fc_lst.append(fc)
        self.fc_lst = nn.ModuleList(self.fc_lst)

        self.last_fc_mean = nn.Linear(current_input_dim, output_dim)
        self.last_fc_logstd = nn.Linear(current_input_dim, output_dim)

        self.activation_function = activation_function

        self.log_std_min = -10
        self.log_std_max = 1

    def forward(self, inputs, deterministic=False, reparam_trick=True, return_log_prob=True):
        h = inputs
        for fc in self.fc_lst:
            h = self.activation_function(fc(h))

        if deterministic:
            return self.get_deterministic_action(h)
        else:
            return self.get_stochastic_action(h, reparam_trick, return_log_prob)

    def get_deterministic_action(self, h):
        actions = torch.tanh(self.last_fc_mean(h))
        log_probs = None

        return actions, log_probs

    def get_stochastic_action(self, h, reparam_trick, return_log_prob):
        mean = self.last_fc_mean(h)
        log_std = self.last_fc_logstd(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        distribution = Normal(mean, std)
        if reparam_trick:
            z = distribution.rsample()
        else:
            z = distribution.sample()
        actions = torch.tanh(z)

        if return_log_prob:
            log_probs = distribution.log_prob(z) - torch.log(1 - actions.pow(2) + 1e-7)
        else:
            log_probs = None

        return actions, log_probs