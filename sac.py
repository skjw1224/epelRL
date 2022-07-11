import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from replay_buffer import ReplayBuffer


class SAC(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        # hyperparameters
        self.h_nodes = self.config.hyperparameters['hidden_nodes']
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.buffer_size = self.config.hyperparameters['buffer_size']
        self.minibatch_size = self.config.hyperparameters['minibatch_size']
        self.crt_learning_rate = self.config.hyperparameters['critic_learning_rate']
        self.act_learning_rate = self.config.hyperparameters['actor_learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.tau = self.config.hyperparameters['tau']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.minibatch_size)

        # Critic-a (+target) net
        self.q_a_net = self.approximator(self.s_dim + self.a_dim, 1, self.h_nodes).to(self.device)
        self.target_q_a_net = self.approximator(self.s_dim + self.a_dim, 1, self.h_nodes).to(self.device)
        for to_model, from_model in zip(self.target_q_a_net.parameters(), self.q_a_net.parameters()):
            to_model.data.copy_(from_model.data.clone())
        self.q_a_net_opt = optim.Adam(self.q_a_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Critic-b (+target) net
        self.q_b_net = self.approximator(self.s_dim + self.a_dim, 1, self.h_nodes).to(self.device)
        self.target_q_b_net = self.approximator(self.s_dim + self.a_dim, 1, self.h_nodes).to(self.device)
        for to_model, from_model in zip(self.target_q_b_net.parameters(), self.q_b_net.parameters()):
            to_model.data.copy_(from_model.data.clone())
        self.q_b_net_opt = optim.Adam(self.q_b_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor net
        self.a_net = self.approximator(self.s_dim, 2 * self.a_dim, self.h_nodes).to(self.device)
        self.a_net_opt = optim.Adam(self.a_net.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Temperature learning
        self.automatic_temp_tuning = self.config.hyperparameters['automatic_temp_tuning']
        if self.automatic_temp_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.a_dim).to(self.device)).item()  # heuristic value from the paper
            self.log_temp = torch.zeros(1, requires_grad=True, device=self.device)
            self.temp = self.log_temp.exp()
            self.temp_optim = optim.Adam([self.log_temp], lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg) # 혹은 따로 지정
        else:
            self.temp = self.config.hyperparameters['temperature']

    def ctrl(self, epi, step, x, u):
        """Picks an action using one of three methods:
        1) Randomly if we haven't passed a certain number of steps,
        2) Using the actor in evaluation mode if eval_ep is True
        3) Using the actor in training mode if eval_ep is False.
        The difference between evaluation and training mode is that training mode does more exploration
        이 action choice 방법을 따라야 할까? 이부분은 꽤나 자유롭게 고를 수 있을것으로 보이는데..."""

        if epi < self.init_ctrl_idx:
            u_nom = self.initial_ctrl.ctrl(epi, step, x, u)
            u_val = self.explorer.sample(epi, step, u_nom)
        else:
            u_val, _, _ = self.sample_action_and_log_prob(x)

        u_val = np.clip(u_val, -1., 1.)

        return u_val

    def add_experience(self, *single_expr):
        x, u, r, x2, is_term = single_expr
        # dfdx, dfdu, dcdx, d2cdu2_inv = derivs
        self.replay_buffer.add(*[x, u, r, x2, is_term])

    def sample_action_and_log_prob(self, x):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action
        NN 이 예측하는것이 평균, 표편이 아니라 평균과 log 표편인듯? """
        x = torch.from_numpy(x.T).float().to(self.device)

        self.a_net.eval()
        with torch.no_grad():
            a_pred = self.a_net(x)
        self.a_net.train()

        mean, log_std = a_pred[:, :self.a_dim], a_pred[:, self.a_dim:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  #rsample means it is sampled using reparameterisation trick
        # tanh 는 action support를 finite로 만들기 위함, 이렇게 나두면 현재는 action 이 -1 에서 1임
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1E-7) # tanh에 따른 미분 값 보정, epsilon은 -inf 방지용
        log_prob = log_prob.sum(1, keepdim=True)

        action = action.T.detach().cpu().numpy()

        return action, log_prob, mean

    def train(self, step):
        def nn_update_one_step(orig_net, target_net, opt, loss):
            """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
            opt.zero_grad()
            loss.backward()
            if orig_net is not None:
                torch.nn.utils.clip_grad_norm_(orig_net.parameters(), self.grad_clip_mag)
            opt.step()

            if target_net is not None:
                """Updates the target network in the direction of the local network but by taking a step size
               less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
                for to_model, from_model in zip(target_net.parameters(), orig_net.parameters()):
                    to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample()

        # Critic Train
        """Calculates the losses for the two critics.
                This is the ordinary Q-learning loss except the additional entropy term is taken into account"""
        with torch.no_grad():
            a2_batch, next_state_log_pi, _ = self.sample_action_and_log_prob(s2_batch.T.detach().cpu().numpy())
            a2_batch = torch.from_numpy(a2_batch.T).float().to(self.device)
            target_q_a2_batch = self.target_q_a_net(torch.cat((s2_batch, a2_batch), dim=-1))
            target_q_b2_batch = self.target_q_b_net(torch.cat((s2_batch, a2_batch), dim=-1))
            max_Q_next_target = torch.max(target_q_a2_batch, target_q_b2_batch) - self.temp * next_state_log_pi
            q_target_batch = r_batch + max_Q_next_target
        q_a_batch = self.q_a_net(torch.cat((s_batch, a_batch), dim=-1))
        q_b_batch = self.q_b_net(torch.cat((s_batch, a_batch), dim=-1))
        q_a_loss = F.mse_loss(q_a_batch, q_target_batch)
        q_b_loss = F.mse_loss(q_b_batch, q_target_batch)

        nn_update_one_step(self.q_a_net, self.target_q_a_net, self.q_a_net_opt, q_a_loss)
        nn_update_one_step(self.q_b_net, self.target_q_b_net, self.q_b_net_opt, q_b_loss)

        # Actor Train
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        a_pred_batch, log_pi, _ = self.sample_action_and_log_prob(s_batch.T.detach().cpu().numpy())
        a_pred_batch = torch.from_numpy(a_pred_batch.T).float().to(self.device)
        q_a_batch = self.q_a_net(torch.cat((s_batch, a_pred_batch), dim=-1))
        q_b_batch = self.q_b_net(torch.cat((s_batch, a_pred_batch), dim=-1))
        Actor_loss = (torch.max(q_a_batch, q_b_batch) - (self.temp * log_pi)).mean()  # Mean은 batch이기 때문

        nn_update_one_step(self.a_net, None, self.a_net_opt, Actor_loss)

        # Temperature parameter Train
        """Calculates the loss for the entropy temperature parameter.
                This is only relevant if self.automatic_entropy_tuning is True."""
        if self.automatic_temp_tuning:
            temp_loss = -(self.log_temp * (log_pi + self.target_entropy).detach()).mean()  # Check sign
            nn_update_one_step(None, None, self.temp_optim, temp_loss)
            self.temp = self.log_temp.exp()

        total_loss = q_a_loss + q_b_loss + Actor_loss
        total_loss = total_loss.detach().cpu().item()

        return total_loss
