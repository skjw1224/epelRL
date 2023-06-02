import torch
import numpy as np
import psutil
import os
from scipy.optimize import fmin_l_bfgs_b

from replay_buffer import ReplayBuffer

MAX_EPOCH = 100
EPSILON = 0.01


class REPS(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        self.epoch = 0

        # hyperparameters
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.max_kl_divergence = self.config.hyperparameters['max_kl_divergence']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT*MAX_EPOCH, batch_size=self.nT*MAX_EPOCH)

        self.phi_dim = 7
        self.phi = RBF(self.s_dim,self.phi_dim,gaussian)
        self.bounds = [(0,None)]*self.end_t+[(None,None)]*(self.end_t*self.phi_dim)
        self.eta = 1 + torch.rand([self.end_t], dtype=torch.float, device=self.device)
        self.theta = torch.rand([self.end_t*self.phi_dim], dtype=torch.float, device=self.device)
        self.st = torch.zeros([self.end_t, self.a_dim], dtype=torch.float, device=self.device)
        self.St = torch.zeros([self.end_t, self.a_dim, self.s_dim], dtype=torch.float, device=self.device)
        self.stdt = 20*torch.eye(self.a_dim, dtype=torch.float, device=self.device).expand_as(torch.zeros([self.end_t, self.a_dim, self.a_dim]))

    def ctrl(self, epi, step, s, a):
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.ctrl(epi, step, s, a)
            a_val = self.explorer.sample(epi, step, a_nom)
        else:
            a_val = self._choose_action(s, step)

        a_val = np.clip(a_val, -1., 1.)

        return a_val

    def _choose_action(self, x, step):
        st = self.st[step]
        St = self.St[step]
        mt = st + torch.mm(St,x.T).squeeze(-1)

        stdt = self.stdt[step]

        mt = mt.cpu().detach().numpy()
        stdt = stdt.cpu().detach().numpy()
        a_nom = np.random.multivariate_normal(mt,stdt)
        return torch.tensor(a_nom, dtype=torch.float, device=self.device).unsqueeze(0)

    def add_experience(self, *single_expr):
        s, a, r, s2, is_term = single_expr
        self.replay_buffer.add(*[s, a, r, s2, is_term])

    def train(self, step):
        if step == self.nT:
            self.policy_update()
            print('policy improved')
            self.epoch = 0  # reset the number of single paths after training
            self.replay_memory.clear()

    def policy_update(self):
        # eta = self.eta.cpu().detach().numpy()
        # theta = self.theta.cpu().detach().numpy().reshape(self.phi_dim*self.end_t)
        # x0 = np.concatenate([eta,theta])
        x0 = torch.cat([self.eta, self.theta]).cpu()
        sol = fmin_l_bfgs_b(self.solve_dualfunc, x0, bounds=self.bounds)
        self.eta = torch.tensor(sol[0][:self.end_t], dtype=torch.float, device=self.device)
        self.theta = torch.tensor(sol[0][self.end_t:], dtype=torch.float, device=self.device)

        self.get_weightedML()

    def solve_dualfunc(self, x):
        x = x.reshape([self.end_t, 1+self.phi_dim])
        x = torch.tensor(x, dtype=torch.float64, device=self.device)
        eta = x[:,0]
        theta = x[:,1:]

        s_batch, a_batch, r_batch, s2_batch, term_batch, _, _, _, _ = self.replay_memory.sample_sequence()

        s_batch = s_batch.double()
        a_batch = a_batch.double()
        r_batch = r_batch.double()
        s2_batch = s2_batch.double()
        term_batch = term_batch.double()

        phi_batch = self.phi(s_batch)
        phi2_batch = self.phi(s2_batch)

        dual = torch.zeros([1], dtype=torch.float64, device=self.device)
        dual_deta = torch.zeros([self.end_t], dtype=torch.float64, device=self.device)
        dual_dtheta = torch.zeros([self.end_t*self.phi_dim], dtype=torch.float64, device=self.device)

        log_vec_prev = torch.ones([MAX_EPOCH], dtype=torch.float64, device=self.device)

        for t in range(self.end_t):
            # log_sum = torch.zeros([1], device=self.device)
            # pt = torch.zeros([MAX_EPOCH])
            # ptj_sum = torch.zeros([1])
            delta = torch.zeros([MAX_EPOCH], dtype=torch.float64, device=self.device)
            d_delta = torch.zeros([MAX_EPOCH, self.s_dim], dtype=torch.float64, device=self.device)
            d_delta_prev = torch.zeros([MAX_EPOCH, self.s_dim], dtype=torch.float64, device=self.device)
            r_batch_t = torch.zeros([MAX_EPOCH], dtype=torch.float64, device=self.device)
            phi = torch.zeros([MAX_EPOCH, self.s_dim], dtype=torch.float64, device=self.device)
            phi2_batch_t = torch.zeros([MAX_EPOCH, self.s_dim], dtype=torch.float64, device=self.device)
            a_batch_t = torch.zeros([MAX_EPOCH, self.a_dim], dtype=torch.float64, device=self.device)
            # term_batch_t = torch.zeros([MAX_EPOCH])

            for j in range(MAX_EPOCH):
                r_batch_t[j] = r_batch[t + j * self.end_t]
                phi[j] = phi_batch[t + j * self.end_t]
                phi2_batch_t[j] = phi2_batch[t + j * self.end_t]
                a_batch_t[j] = a_batch[t + j * self.end_t]
                # term_batch_t[j] = term_batch[t+j*self.end_t]

                if term_batch[t + j * self.end_t]:
                    delta[j] = r_batch[t + j * self.end_t] - torch.matmul(phi_batch[t + j * self.end_t], theta[t].T)
                else:
                    delta[j] = r_batch[t + j * self.end_t] + torch.matmul(phi2_batch[t + j * self.end_t], theta[t + 1].T) \
                               - torch.matmul(phi_batch[t + j * self.end_t], theta[t].T)

                d_delta[j] = -phi[j]
                d_delta_prev[j] = phi[j]

            pt = torch.exp(delta / eta[t])
            pt /= torch.sum(pt)

            log_vec = pt * torch.exp(EPSILON + delta / eta[t])
            log_sum = torch.sum(log_vec)
            dual = dual + eta[t] * torch.log(log_sum)

            log_vec_eta = log_vec * delta
            log_sum_eta = torch.sum(log_vec_eta)
            dual_deta[t] = torch.log(log_sum) - (1/eta[t])**2*log_sum_eta/log_sum

            log_sum_theta = torch.matmul(log_vec,d_delta)

            log_sum_prev = torch.sum(log_vec_prev)
            log_sum_theta_prev = torch.matmul(log_vec_prev,d_delta_prev)
            dual_dtheta[t*self.phi_dim:(t+1)*self.phi_dim] = log_sum_theta/log_sum + log_sum_theta_prev/log_sum_prev

            log_vec_prev = log_vec

        dual = dual.cpu().detach().numpy()
        d_dual = torch.cat([dual_deta,dual_dtheta])
        d_dual = d_dual.cpu().detach().numpy()

        return dual, d_dual

    def get_weightedML(self):

        s_batch, a_batch, r_batch, s2_batch, term_batch, _, _, _, _ = self.replay_memory.sample_sequence()

        for t in range(self.end_t):
            # pt = torch.zeros([MAX_EPOCH])
            # ptj_sum = torch.zeros([1])
            delta = torch.zeros([MAX_EPOCH], dtype=torch.float, device=self.device)
            Xt = torch.ones([MAX_EPOCH,1+self.s_dim], dtype=torch.float, device=self.device)
            Ut = torch.zeros([MAX_EPOCH,self.a_dim], dtype=torch.float, device=self.device)

            r_batch_t = torch.zeros([MAX_EPOCH], dtype=torch.float, device=self.device)
            s_batch_t = torch.zeros([MAX_EPOCH,self.s_dim], dtype=torch.float, device=self.device)
            s2_batch_t = torch.zeros([MAX_EPOCH,self.s_dim], dtype=torch.float, device=self.device)
            a_batch_t = torch.zeros([MAX_EPOCH,self.a_dim], dtype=torch.float, device=self.device)
            # term_batch_t = torch.zeros([MAX_EPOCH])

            for j in range(MAX_EPOCH):
                r_batch_t[j] = r_batch[t+j*self.end_t]
                s_batch_t[j] = s_batch[t+j*self.end_t]
                s2_batch_t[j] = s2_batch[t+j*self.end_t]
                a_batch_t[j] = a_batch[t+j*self.end_t]
                # term_batch_t[j] = term_batch[t+j*self.end_t]

                if term_batch[t+j*self.end_t]:
                    delta[j] = r_batch[t+j*self.end_t] - torch.matmul(s_batch[t+j*self.end_t],self.theta[t*self.phi_dim:(t+1)*self.phi_dim].T)
                else:
                    delta[j] = r_batch[t+j*self.end_t] + torch.matmul(s2_batch[t+j*self.end_t],self.theta[t*self.phi_dim:(t+1)*self.phi_dim].T) \
                            - torch.matmul(s_batch[t+j*self.end_t],self.theta[t*self.phi_dim:(t+1)*self.phi_dim].T)

            pt = torch.exp(delta/self.eta[t])
            pt /= torch.sum(pt)

            Xt[:,1:] = s_batch_t
            Ut[:,:] = a_batch_t
            Dt = torch.diag(pt)

            XtDtXt = torch.mm(torch.mm(Xt.T,Dt),Xt)
            XtDtUt = torch.mm(torch.mm(Xt.T, Dt), Ut)
            weightedML = torch.mm(torch.inverse(XtDtXt),XtDtUt)
            self.st[t] = weightedML[0].T
            self.St[t] = weightedML[1:].T

            mut = torch.zeros([MAX_EPOCH,self.a_dim], dtype=torch.float, device=self.device )
            for k in range(MAX_EPOCH):
                mut[k] = self.st[t] + torch.matmul(self.St[t],s_batch_t[k])

            er = mut - a_batch_t
            self.stdt[t] = torch.mm(torch.mm(er.T, Dt), er)/torch.sum(pt)

        # del s_batch, a_batch, r_batch, s2_batch, term_batch

    def check_memory_usage(self,when):
        if when == 'before':
            # general RAM usage
            memory_usage_dict = dict(psutil.virtual_memory()._asdict())
            memory_usage_percent = memory_usage_dict['percent']
            print(f"BEFORE CODE: memory_usage_percent: {memory_usage_percent}%")
            # current process RAM usage
            pid = os.getpid()
            current_process = psutil.Process(pid)
            current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
            print(f"BEFORE CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")
        else:
            # AFTER  code
            memory_usage_dict = dict(psutil.virtual_memory()._asdict())
            memory_usage_percent = memory_usage_dict['percent']
            print(f"AFTER  CODE: memory_usage_percent: {memory_usage_percent}%")
            # current process RAM usage
            pid = os.getpid()
            current_process = psutil.Process(pid)
            current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2. ** 20
            print(f"AFTER  CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")
