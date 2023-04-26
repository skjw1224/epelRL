import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

from replay_buffer import ReplayBuffer


class TRPO(object):
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
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']
        self.grad_clip_mag = self.config.hyperparameters['grad_clip_mag']
        self.tau = self.config.hyperparameters['tau']
        self.gae_lambda = self.config.hyperparameters['gae_lambda']
        self.gae_gamma = self.config.hyperparameters['gae_gamma']
        self.num_critic_update = self.config.hyperparameters['num_critic_update']
        self.max_kl = self.config.hyperparameters['max_kl']

        self.epoch = 0  # for single path sampling

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.minibatch_size)

        # Policy network
        self.actor_net = self.approximator(self.s_dim, self.a_dim, self.h_nodes).to(self.device)
        self.actor_net.action_log_std = nn.Parameter(torch.zeros(1, self.a_dim))
        self.old_actor_net = self.approximator(self.s_dim, self.a_dim, self.h_nodes).to(self.device)

        # Value network
        self.critic_net = self.approximator(self.s_dim, 1, self.h_nodes).to(self.device)
        self.critic_net_opt = optim.Adam(self.critic_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

    def ctrl(self, epi, step, s, a):
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.ctrl(epi, step, s, a)
            a_val = self.explorer.sample(epi, step, a_nom)
        else:
            a_val = self._choose_action(s)

        # if self.epoch == MAX_EPOCH:  # train the network with MAX_EPOCH single paths
        #     self.nn_train()
        #     self.epoch = 0  # reset the number of single paths after training
        #     self.epsilon = EPSILON / (1. + (epi / EPI_DENOM))  # instead of using "exp_schedule" method # (epi % MAX_EPOCH)?

        return a_val

    def _choose_action(self, s):
        # numpy to torch
        s = torch.from_numpy(s.T).float().to(self.device)

        self.actor_net.eval()
        with torch.no_grad():
            a = self.actor_net(s)
        self.actor_net.train()

        # torch to numpy
        a = a.T.cpu().detach().numpy()

        return a

    def add_experience(self, *single_expr):
        s, a, r, s2, is_term = single_expr
        self.replay_buffer.add(*[s, a, r, s2, is_term])

        if is_term:  # count the number of single paths for one training
            self.epoch += 1

    def train(self, step):
        # Replay buffer sample
        s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample_sequence()

        # step 1: get returns and GAEs
        returns, advantages = self._gae_estimation(s_batch, r_batch, term_batch)

        # step 2: train critic network several steps with respect to returns
        self._critic_update(s_batch, returns, advantages)

        # step 3: get gradient of loss and hessian of kl
        mean = self.actor_net(s_batch)
        log_std = self.actor_net.action_log_std.expand_as(mean).to(self.device)
        std = torch.exp(log_std).to(self.device)
        old_policy = self.log_density(a_batch, mean, std, log_std)

        loss = self._surrogate_loss(advantages, s_batch.detach(), old_policy.detach(), a_batch.detach())
        loss_grad = torch.autograd.grad(loss, self.actor_net.parameters())
        loss_grad = self._flat_grad(loss_grad)
        step_dir = self.conjugate_gradient(self.actor_net, s_batch, loss_grad.data, nsteps=10)

        # step 4: get step direction and step size and full step
        params = self.flat_params(self.actor_net)
        shs = 0.5 * (step_dir * self.fisher_vector_product(self.actor_net, s_batch, step_dir)).sum(0, keepdim=True)
        step_size = 1 / torch.sqrt(shs / self.max_kl)[0]
        full_step = -step_size * step_dir

        # step 5: do backtracking line search for n times
        self.update_model(self.old_actor_net, params)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)

        flag = False
        fraction = 1.0
        for i in range(10):
            new_params = params + fraction * full_step
            self.update_model(self.actor_net, new_params)
            new_loss = self._surrogate_loss(advantages, s_batch, old_policy.detach(), a_batch)
            loss_improve = new_loss - loss
            expected_improve *= fraction
            kl = self.kl_divergence(new_actor=self.actor_net, old_actor=self.old_actor_net, states=s_batch)
            kl = kl.mean()

            print('kl: {:.4f}  loss improve: {:.4f}  expected improve: {:.4f}  '
                  'number of line search: {}'
                  .format(kl.data.numpy(), loss_improve, expected_improve[0], i))

            # see https: // en.wikipedia.org / wiki / Backtracking_line_search
            if kl < self.max_kl and (loss_improve / expected_improve) > 0.5:
                flag = True
                break

            fraction *= 0.5

        if not flag:
            params = self.flat_params(self.old_actor_net)
            self.update_model(self.actor_net, params)
            print('policy update does not impove the surrogate')

    def _gae_estimation(self, states, rewards, terms):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        values = self.critic_net(states)

        for t in reversed(range(len(rewards))):
            prev_return = rewards[t] + self.gae_gamma * prev_return * (1 - terms[t])
            td_error = rewards[t] + self.gae_gamma * prev_value * (1 - terms[t]) - values.data[t]
            prev_advantage = td_error + self.gae_gamma * self.gae_lambda * prev_advantage * (1 - terms[t])
            prev_value = values.data[t]

            returns[t] = prev_return
            advantages[t] = prev_advantage

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def _critic_update(self, states, returns, advantages):
        criterion = nn.MSELoss()
        n = len(states)
        arr = np.arange(n)

        for _ in range(self.num_critic_update):
            np.random.shuffle(arr)

            for i in range(n // self.minibatch_size):
                batch_index = torch.LongTensor(arr[self.minibatch_size * i: self.minibatch_size * (i + 1)])
                values = self.critic_net(states[batch_index])
                target = returns.unsqueeze(1)[batch_index] + advantages.unsqueeze(1)[batch_index]

                loss = criterion(values, target)
                self.critic_net_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.grad_clip_mag)
                self.critic_net_optim.step()

    def _surrogate_loss(self, advantages, states, old_policy, actions):
        mean = self.actor_net(states)
        log_std = self.actor_net.action_log_std.expand_as(mean).to(self.device)
        std = torch.exp(log_std).to(self.device)

        new_policy = self.log_density(actions, mean, std, log_std)
        advantages = advantages.unsqueeze(1)

        surrogate_loss = advantages * torch.exp(new_policy - old_policy)
        surrogate_loss = surrogate_loss.mean()

        return surrogate_loss

    def log_density(self, a, mu, std, logstd):
        var = std.pow(2)
        log_density = -(a - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
        return log_density.sum(1, keepdim=True)

    def _flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)

        return grad_flatten

    def flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten

    def flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten

    def update_model(self, model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length

    def kl_divergence(self, new_actor, old_actor, states):
        mu, std, logstd = new_actor(states.clone().detach().requires_grad_(True))
        mu_old, std_old, logstd_old = old_actor(
            states.clone().detach().requires_grad_(True))  # requires_grad == False?
        mu_old = mu_old.detach()
        std_old = std_old.detach()
        logstd_old = logstd_old.detach()

        # kl divergence between old policy and new policy : D( pi_old || pi_new )
        # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
        # be careful of calculating KL-divergence. It is not symmetric metric
        kl = logstd - logstd_old + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
             (2.0 * std.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def fisher_vector_product(self, actor, states, p):
        p.detach()
        kl = self.kl_divergence(new_actor=actor, old_actor=actor, states=states)
        kl = kl.mean()
        kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
        kl_grad = self.flat_grad(kl_grad)  # check kl_grad == 0

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters())
        kl_hessian_p = self.flat_hessian(kl_hessian_p)

        return kl_hessian_p + 0.1 * p

    # from openai baseline code
    # https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    def conjugate_gradient(self, actor, states, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self.fisher_vector_product(actor, states, p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x
