import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import namedtuple

from replay_buffer import ReplayBuffer
from explorers import OU_Noise


GAMMA = 1
LAMBDA = 0.98
MAX_KL = 0.01
# clip_param = 0.2
MAX_EPOCH = 10

BUFFER_SIZE = 180*MAX_EPOCH
MINIBATCH_SIZE = 64
TAU = 0.05  # for GAE
EPSILON = 0.99
EPI_DENOM = 1.

CRT_LEARNING_RATE = 0.0003
ACT_LEARNING_RATE = 0.0003
ADAM_EPS = 1E-4
L2REG = 1E-3
GRAD_CLIP = 10.0

INITIAL_POLICY_INDEX = 10
AC_PE_TRAINING_INDEX = 20


class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a deep Q-learning network
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(Network, self).__init__()
        n_h_nodes = [64, 64, 64]

        self.fc1 = nn.Linear(input_dim, n_h_nodes[0])
        self.fc2 = nn.Linear(n_h_nodes[0], n_h_nodes[1])
        self.fc3 = nn.Linear(n_h_nodes[2], output_dim)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


class TRPO:
    def __init__(self, env, device):
        self.end_t = env.nT
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.epoch = 0  # for single path sampling
        self.epsilon = .1
        self.a_exp_history = torch.tensor([],device=device)

        self.device = device

        self.replay_memory = ReplayBuffer(env, device, self.end_t*MAX_EPOCH, self.end_t*MAX_EPOCH)
        self.exp_noise = OU_Noise(self.a_dim)
        self.initial_ctrl = InitialControl(env, device)

        # Policy (+old) net
        self.p_net = Network(self.s_dim,self.a_dim).to(device)
        self.old_p_net = Network(self.s_dim, self.a_dim).to(device)

        # Value net
        self.v_net = Network(self.s_dim,1).to(device)
        self.v_net_opt = torch.optim.Adam(self.v_net.parameters(), lr=CRT_LEARNING_RATE, eps=ADAM_EPS, weight_decay=L2REG)

    def ctrl(self, epi, step, x, u):

        a_exp = self.exp_schedule(epi, step)
        self.a_exp_history = torch.cat((self.a_exp_history, a_exp.unsqueeze(0)))
        if epi < INITIAL_POLICY_INDEX:
            a_nom = self.initial_ctrl.controller(step, x, u)
            self.epoch = 0
            self.replay_memory.clear()
            self.a_exp_history = torch.tensor([], device=self.device)
        elif INITIAL_POLICY_INDEX <= epi < AC_PE_TRAINING_INDEX:
            if INITIAL_POLICY_INDEX == epi and step == 0:
                self.epoch = 0
                self.replay_memory.clear()
            a_nom = self.choose_action(x)
        else:
            a_nom = self.choose_action(x)

        # a_val = torch.normal(a_nom, torch.exp(a_exp))
        a_val = torch.normal(a_nom, a_exp)

        return a_val.detach()

    def choose_action(self, s):
        self.p_net.eval()
        with torch.no_grad():
            a_nom = self.p_net(s)
        self.p_net.train()
        return a_nom

    def add_experience(self, *single_expr):
        x, u, r, x2, term, derivs = single_expr
        self.replay_memory.add(*[x, u, r, x2, term, *derivs])

        if term:  # count the number of single paths for one training
            self.epoch += 1

        if self.epoch == MAX_EPOCH:  # train the network with MAX_EPOCH single paths
            self.nn_train()
            print('policy improved')
            self.epoch = 0  # reset the number of single paths after training
            self.replay_memory.clear()
            self.a_exp_history = torch.tensor([], device=self.device)

    def exp_schedule(self, epi, step):
        noise = self.exp_noise.sample() / 10.
        if epi % MAX_EPOCH == 0:
            self.epsilon = EPSILON / (1. + (epi / EPI_DENOM))
        a_exp = noise * self.epsilon
        return torch.tensor(a_exp, dtype=torch.float, device=self.device)

    def nn_train(self):

        # Replay buffer sample
        s_batch, a_batch, r_batch, s2_batch, term_batch, _, _, _, _ = self.replay_memory.sample_sequence()

        # ----------------------------
        # step 1: get returns and GAEs
        values = self.v_net(s_batch.clone().detach().requires_grad_(True))
        returns, advants = self.get_gae(r_batch, term_batch, values)

        # ----------------------------
        # step 2: train v_net several steps with respect to returns
        self.nn_update_v_net(self.v_net, s_batch, returns, advants, self.v_net_opt)

        # ----------------------------
        # step 3: get gradient of loss and hessian of kl
        mu = self.p_net(s_batch)
        logstd = self.a_exp_history
        std = torch.exp(logstd)
        old_policy = self.log_density(a_batch, mu, std, logstd)

        loss = self.surrogate_loss(self.p_net, advants, s_batch.detach(), old_policy.detach(), a_batch.detach())
        loss_grad = torch.autograd.grad(loss, self.p_net.parameters())
        loss_grad = self.flat_grad(loss_grad)
        step_dir = self.conjugate_gradient(self.p_net, s_batch, loss_grad.data, nsteps=10)

        # ----------------------------
        # step 4: get step direction and step size and full step
        params = self.flat_params(self.p_net)
        shs = 0.5 * (step_dir * self.fisher_vector_product(self.p_net, s_batch, step_dir)
                     ).sum(0, keepdim=True)
        step_size = 1 / torch.sqrt(shs / MAX_KL)[0]
        full_step = -step_size * step_dir

        # ----------------------------
        # step 5: do backtracking line search for n times
        self.update_model(self.old_p_net, params)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)

        flag = False
        fraction = 1.0
        for i in range(10):
            new_params = params + fraction * full_step
            self.update_model(self.p_net, new_params)
            new_loss = self.surrogate_loss(self.p_net, advants, s_batch, old_policy.detach(),
                                           a_batch)
            loss_improve = new_loss - loss
            expected_improve *= fraction
            kl = self.kl_divergence(new_actor=self.p_net, old_actor=self.old_p_net, states=s_batch)
            kl = kl.mean()

            print('kl: {:.4f}  loss improve: {:.4f}  expected improve: {:.4f}  '
                  'number of line search: {}'
                  .format(kl.data.cpu().numpy(), loss_improve, expected_improve[0], i))

            # see https: // en.wikipedia.org / wiki / Backtracking_line_search
            if kl < MAX_KL and (loss_improve / expected_improve) > 0.5:
                flag = True
                break

            fraction *= 0.5

        if not flag:
            params = self.flat_params(self.old_p_net)
            self.update_model(self.p_net, params)
            print('policy update does not impove the surrogate')

    # utils for TRPO
    def get_gae(self, rewards, terms, values):
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)

        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            if terms[t]:
                mask = 0
            else:
                mask = 1
            running_returns = rewards[t] + GAMMA * running_returns * mask
            running_tderror = rewards[t] + GAMMA * previous_value * mask - \
                              values.data[t]
            running_advants = running_tderror + GAMMA * LAMBDA * \
                              running_advants * mask

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants

        advants = (advants - advants.mean()) / advants.std()
        return returns.to(self.device), advants.to(self.device)

    def surrogate_loss(self, actor, advants, states, old_policy, actions):
        mu = actor(states)
        logstd = self.a_exp_history
        std = torch.exp(logstd)
        new_policy = self.log_density(actions, mu, std, logstd)
        advants = advants.unsqueeze(1)

        surrogate = advants * torch.exp(new_policy - old_policy)
        surrogate = surrogate.mean()
        return surrogate

    def nn_update_v_net(self, critic, states, returns, advants, critic_optim):
        criterion = torch.nn.MSELoss()
        n = len(states)
        arr = np.arange(n)

        for epoch in range(5):
            np.random.shuffle(arr)

            for i in range(n // MINIBATCH_SIZE):
                batch_index = arr[MINIBATCH_SIZE * i: MINIBATCH_SIZE * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                inputs = states[batch_index]
                target1 = returns[batch_index].clone().detach()
                target2 = advants[batch_index].clone().detach()

                values = critic(inputs.clone().detach().requires_grad_(True))
                loss = criterion(values, target1 + target2)
                critic_optim.zero_grad()
                loss.backward()
                critic_optim.step()

    def log_density(self, a, mu, std, logstd):
        var = std.pow(2)
        log_density = -(a - mu).pow(2) / (2 * var) \
                      - 0.5 * math.log(2 * math.pi) - logstd
        return log_density.sum(1, keepdim=True)

    def flat_grad(self, grads):
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
        mu = new_actor(states.clone().detach().requires_grad_(True))
        mu_old = old_actor(states.clone().detach())
        logstd = self.a_exp_history
        std = torch.exp(logstd)

        std_old = std
        logstd_old = logstd

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
        x = torch.zeros(b.size()).to(self.device)
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


from ilqr import ILQR
class InitialControl(object):
    def __init__(self, env, device):
        self.ilqr = ILQR(env, device)

    def controller(self, step, x, u):
        return self.ilqr.lqr_ref(step, x, u)