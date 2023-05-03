import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import copy

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
        self.num_cg_iterations = self.config.hyperparameters['num_cg_iterations']
        self.max_kl_divergence = self.config.hyperparameters['max_kl_divergence']

        self.epoch = 0  # for single path sampling

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT, batch_size=self.nT)

        # Policy network
        self.actor_net = self.approximator(self.s_dim, 2 * self.a_dim, self.h_nodes).to(self.device)
        self.old_actor_net = self.approximator(self.s_dim, 2 * self.a_dim, self.h_nodes).to(self.device)

        # Value network
        self.critic_net = self.approximator(self.s_dim, 1, self.h_nodes).to(self.device)
        self.critic_net_opt = optim.Adam(self.critic_net.parameters(), lr=self.crt_learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

    def ctrl(self, epi, step, s, a):
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.ctrl(epi, step, s, a)
            a_val = self.explorer.sample(epi, step, a_nom)
        else:
            a_val = self._choose_action(s)

        a_val = np.clip(a_val, -1., 1.)

        return a_val

    def _choose_action(self, s):
        # numpy to torch
        s = torch.from_numpy(s.T).float().to(self.device)

        self.actor_net.eval()
        with torch.no_grad():
            a_pred = self.actor_net(s)
        self.actor_net.train()

        mean, log_std = a_pred[:, :self.a_dim], a_pred[:, self.a_dim:]
        std = log_std.exp()
        a_distribution = Normal(mean, std)
        a = a_distribution.sample()
        a = torch.tanh(a)

        # torch to numpy
        a = a.T.cpu().detach().numpy()

        return a

    def add_experience(self, *single_expr):
        s, a, r, s2, is_term = single_expr
        self.replay_buffer.add(*[s, a, r, s2, is_term])

        if is_term:  # count the number of single paths for one training
            self.epoch += 1

    def train(self, step):
        if step == self.nT - 1:
            # Replay buffer sample
            s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample_sequence()

            # Compute returns and advantages
            returns, advantages = self._gae_estimation(s_batch, r_batch, term_batch)

            # Compute the gradient of surrgoate loss and kl divergence
            loss, kl_div = self._compute_surrogate_loss_and_kl_divergence(s_batch, a_batch, advantages)
            loss_grad = self._flat_gradient(loss, self.actor_net.parameters(), retain_graph=True)
            kl_div_grad = self._flat_gradient(kl_div, self.actor_net.parameters(), create_graph=True)

            # Compute a search direction using the conjugate gradient algorithm
            search_direction = self._conjugate_gradient(kl_div_grad, loss_grad.data, cg_iterations=self.num_cg_iterations)

            # Compute max step size and max step
            sHs = torch.matmul(search_direction, self._hessian_vector_product(kl_div_grad, search_direction))
            max_step_size = torch.sqrt(2 * self.max_kl_divergence / sHs)
            max_step = - max_step_size * search_direction

            # do backtracking line search for n times
            params = self.flat_params(self.actor_net)
            self.update_model(self.old_actor_net, params)
            expected_improve = (loss_grad * max_step).sum(0, keepdim=True)

            flag = False
            fraction = 1.0
            for i in range(10):
                new_params = params + fraction * max_step
                self.update_model(self.actor_net, new_params)
                new_loss = self._surrogate_loss(advantages, s_batch, self.old_actor_net.detach(), a_batch)
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

            # train critic network several steps with respect to returns
            self._critic_update(s_batch, returns, advantages)

        else:
            loss = 0.

        return loss

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
                self.critic_net_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.grad_clip_mag)
                self.critic_net_opt.step()

    def _compute_surrogate_loss_and_kl_divergence(self, states, actions, advantages):
        log_probs_new, distribution_new = self._get_log_probs(states, actions)
        log_probs_old, distribution_old = self._get_log_probs(states, actions)

        surrogate_loss = advantages * torch.exp(log_probs_new - log_probs_old.detach())
        surrogate_loss = surrogate_loss.mean()

        kl_div = torch.distributions.kl_divergence(distribution_old, distribution_new).mean()

        return surrogate_loss, kl_div

    def _get_log_probs(self, states, actions):
        actor_output = self.actor_net(states)
        means, log_stds = actor_output[:, :self.a_dim], actor_output[:, self.a_dim:]
        stds = torch.exp(log_stds)
        distribution = Normal(means, stds)
        log_probs = distribution.log_prob(actions)

        return log_probs, distribution

    def _flat_gradient(self, tensor, parameters, retain_graph=False, create_graph=False):
        if create_graph:
            retain_graph = True

        grads = torch.autograd.grad(tensor, parameters, retain_graph=retain_graph, create_graph=create_graph)
        flat_grad = torch.cat([grad.view(-1) for grad in grads])

        return flat_grad

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

    # from openai baseline code
    # https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    def _conjugate_gradient(self, kl_div_grad, loss_grad, cg_iterations, residual_tol=1e-10):
        x = torch.zeros_like(loss_grad)
        r = loss_grad.clone()
        p = loss_grad.clone()
        rho = torch.dot(r, r)
        for _ in range(cg_iterations):
            hvp = self._hessian_vector_product(kl_div_grad, p)
            alpha = rho / torch.dot(p, hvp)
            x += alpha * p
            r -= alpha * hvp
            rho_new = torch.dot(r, r)
            p = r + (rho_new / rho) * p

            rho = rho_new
            if rho < residual_tol:
                break

        return x

    def _hessian_vector_product(self, kl_div_grad, p):
        vp = torch.matmul(kl_div_grad, p.detach())
        hvp = self._flat_gradient(vp, self.actor_net.parameters(), retain_graph=True)

        return hvp + 0.1 * p
