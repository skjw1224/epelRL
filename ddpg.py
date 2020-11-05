import torch
import torch.nn.functional as F
from replay_buffer import ReplayBuffer


class DDPG(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim

        # hyperparameters
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.explore_epi_idx = self.config.hyperparameters['explore_epi_idx']
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

        # Critic (+target) net
        self.q_net = self.approximator(config, self.s_dim + self.a_dim, 1).to(self.device)
        self.target_q_net = self.approximator(config, self.s_dim + self.a_dim, 1).to(self.device)

        for to_model, from_model in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.q_net_opt = torch.optim.Adam(self.q_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Actor (+target) net
        self.mu_net = self.approximator(config, self.s_dim, self.a_dim).to(self.device)
        self.target_mu_net = self.approximator(config, self.s_dim, self.a_dim).to(self.device)

        for to_model, from_model in zip(self.target_mu_net.parameters(), self.mu_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.mu_net_opt = torch.optim.Adam(self.mu_net.parameters(), lr=self.act_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

    def ctrl(self, epi, step, x, u):
        if epi < self.init_ctrl_idx:
            u_nom = self.initial_ctrl.ctrl(epi, step, x, u)
            u_val = self.explorer.sample(epi, step, u_nom)
        elif self.init_ctrl_idx <= epi < self.explore_epi_idx:
            u_nom = self.choose_action(epi, step, x, u)
            u_val = self.explorer.sample(epi, step, u_nom)
        else:
            u_val = self.choose_action(epi, step, x, u)

        return u_val

    def choose_action(self, epi, step, x, u):
        # Numpy to torch
        x = torch.from_numpy(x.T).float().to(self.device)  # (B, 1)

        # Option: target_actor_net OR actor_net?
        self.mu_net.eval()
        with torch.no_grad():
            u = self.mu_net(x)
        self.mu_net.train()

        # Torch to Numpy
        u = u.T.detach().numpy()
        return u

    def add_experience(self, *single_expr):
        x, u, r, x2, is_term = single_expr
        self.replay_buffer.add(*[x, u, r, x2, is_term])

    def train(self, step):
        def nn_update_one_step(orig_net, target_net, opt, loss):
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(orig_net.parameters(), self.grad_clip_mag)
            opt.step()

            """Updates the target network in the direction of the local network but by taking a step size
           less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
            for to_model, from_model in zip(target_net.parameters(), orig_net.parameters()):
                to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

        if len(self.replay_buffer) > 0:
            # Replay buffer sample
            s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample()

            # Critic Train
            q_batch = self.q_net(torch.cat([s_batch, a_batch], dim=-1))
            a2_batch = self.target_mu_net(s2_batch)

            q_target_batch = r_batch + self.q_net(torch.cat([s2_batch, a2_batch], dim=-1)) * (1 - term_batch)
            q_loss = F.mse_loss(q_batch, q_target_batch)

            nn_update_one_step(self.q_net, self.target_q_net, self.q_net_opt, q_loss)

            # Actor Train
            a_pred_batch = self.mu_net(s_batch)
            a_loss = self.target_q_net(torch.cat([s_batch, a_pred_batch], dim=-1)).mean()
            nn_update_one_step(self.mu_net, self.target_mu_net, self.mu_net_opt, a_loss)

            q_loss = q_loss.detach().numpy().item()
            a_loss = a_loss.detach().numpy().item()
            loss = q_loss + a_loss
        else:
            loss = 0.

        return loss