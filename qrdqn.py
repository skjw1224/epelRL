import torch
from dqn import DQN


class QRDQN(DQN):
    def __init__(self, config):
        DQN.__init__(self, config)

        # Hyperparameters
        self.n_quantiles = self.config.hyperparameters['n_quantiles']

        # Critic network
        self.critic_net = self.approximator(self.s_dim, self.a_dim * self.n_quantiles, self.h_nodes).to(self.device)  # s --> a
        self.target_critic_net = self.approximator(self.s_dim, self.a_dim * self.n_quantiles, self.h_nodes).to(self.device) # s --> a

        for to_model, from_model in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            to_model.data.copy_(from_model.data.clone())

        self.critic_net_opt = torch.optim.Adam(self.critic_net.parameters(), lr=self.learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        self.quantile_taus = ((2 * torch.arange(self.n_quantiles) + 1) / (2. * self.n_quantiles)).unsqueeze(0).to(self.device)
        self.prev_a_idx = None

    def choose_action(self, epi, step, s, a):
        # numpy to torch
        s = torch.from_numpy(s.T).float().to(self.device)  # (B, 1)

        self.critic_net.eval()
        with torch.no_grad():
            a_idx = self.get_value_distribution(self.critic_net, s).mean(2).min(1)[1].unsqueeze(1)
        self.critic_net.train()

        # torch to Numpy
        a_idx = a_idx.detach().cpu().numpy()

        return a_idx

    def train(self):
        if len(self.replay_buffer) > 0:
            s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample()

            q_distribution = self.get_value_distribution(self.critic_net, s_batch)
            q_batch = q_distribution.gather(1, a_batch.unsqueeze(-1).repeat(1, 1, self.n_quantiles).long())

            q2_distribution = self.get_value_distribution(self.target_critic_net, s2_batch, False)
            u_max_idx_batch = q2_distribution.mean(2).min(1)[1].unsqueeze(1)
            q2_batch = q2_distribution.gather(1, u_max_idx_batch.unsqueeze(-1).repeat(1, 1, self.n_quantiles).long())
            q_target_batch = r_batch.unsqueeze(2) + -(-1 + term_batch.unsqueeze(2).float()) * q2_batch

            q_loss = self.quantile_huber_loss(q_batch, q_target_batch)

            self.critic_net_opt.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.grad_clip_mag)
            self.critic_net_opt.step()

            for to_model, from_model in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
                to_model.data.copy_(self.tau * from_model.data + (1 - self.tau) * to_model.data)

            q_loss = q_loss.cpu().detach().numpy().item()
        else:
            q_loss = 0.

        return q_loss

    def get_value_distribution(self, net, s, stack_graph=True):
        if stack_graph:
            net_value = net(s)
        else:
            net_value = net(s).detach()
        return net_value.view(-1, self.a_dim, self.n_quantiles)

    def quantile_huber_loss(self, q_batch, q_target_batch):
        qh_loss_batch = torch.tensor(0., device=self.device)
        huber_loss_fnc = torch.nn.SmoothL1Loss(reduction='none')
        for n, q in enumerate(q_batch):
            q_target = q_target_batch[n]
            error = q_target - q.transpose(0,1)
            huber_loss = huber_loss_fnc(error, torch.zeros(error.shape, device=self.device))
            qh_loss = (huber_loss * (self.quantile_taus - (error < 0).float()).abs()).mean(1).sum(0)
            qh_loss_batch = qh_loss_batch + qh_loss

        return qh_loss_batch
