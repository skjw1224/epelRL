import torch
import numpy as np
import psutil
import os
import sys
from scipy.optimize import fmin_l_bfgs_b
from torch_rbf import *

from explorers import OU_Noise
from replay_buffer import ReplayBuffer

GAMMA = 1
LAMBDA = 0.98
MAX_KL = 0.01
MAX_EPOCH = 100

EPSILON = 0.01
EPI_DENOM = 1.

INITIAL_POLICY_INDEX = 10
AC_PE_TRAINING_INDEX = 10


class REPS:
    def __init__(self,env,device):
        self.end_t = env.nT
        self.s_dim = env.s_dim
        self.a_dim = env.a_dim
        self.epoch = 0

        self.device = device

        self.replay_memory = ReplayBuffer(env, device, self.end_t*MAX_EPOCH, self.end_t*MAX_EPOCH)
        self.exp_noise = OU_Noise(self.a_dim)
        self.initial_ctrl = InitialControl(env, device)

        self.phi_dim = 7
        self.phi = RBF(self.s_dim,self.phi_dim,gaussian)
        self.bounds = [(0,None)]*self.end_t+[(None,None)]*(self.end_t*self.phi_dim)
        self.eta = 1 + torch.rand([self.end_t], dtype=torch.float, device=self.device)
        self.theta = torch.rand([self.end_t*self.phi_dim], dtype=torch.float, device=self.device)
        self.st = torch.zeros([self.end_t, self.a_dim], dtype=torch.float, device=self.device)
        self.St = torch.zeros([self.end_t, self.a_dim, self.s_dim], dtype=torch.float, device=self.device)
        self.stdt = 20*torch.eye(self.a_dim, dtype=torch.float, device=self.device).expand_as(torch.zeros([self.end_t, self.a_dim, self.a_dim]))

    def ctrl(self, epi, step, *single_expr):
        x, u, r, x2, term, derivs = single_expr
        x = x + 0.001*torch.abs(torch.rand([1,self.s_dim], dtype=torch.float, device=self.device))
        x2 = x2 + 0.001 * torch.abs(torch.rand([1, self.s_dim], dtype=torch.float, device=self.device))
        self.replay_memory.add(*[x, u, r, x2, term, *derivs])

        if term:  # count the number of single paths for one training
            self.epoch += 1

        # self.a_exp_history = torch.cat((self.a_exp_history, a_exp.unsqueeze(0)))
        if epi < INITIAL_POLICY_INDEX:
            a_exp = self.exp_schedule(epi, step)
            a_nom = self.initial_ctrl.controller(step, x, u)
            a_val = a_nom + .1*a_exp
        elif INITIAL_POLICY_INDEX <= epi < AC_PE_TRAINING_INDEX:
            a_val = self.choose_action(x, step)
        else:
            a_val = self.choose_action(x, step)

        if self.epoch == MAX_EPOCH:  # train the network with MAX_EPOCH single paths
            self.policy_update()
            print('policy improved')
            self.epoch = 0  # reset the number of single paths after training
            self.replay_memory.clear()
        return a_val.detach()

    def choose_action(self, x, step):
        st = self.st[step]
        St = self.St[step]
        mt = st + torch.mm(St,x.T).squeeze(-1)

        stdt = self.stdt[step]

        mt = mt.cpu().detach().numpy()
        stdt = stdt.cpu().detach().numpy()
        a_nom = np.random.multivariate_normal(mt,stdt)
        return torch.tensor(a_nom, dtype=torch.float, device=self.device).unsqueeze(0)

    def exp_schedule(self, epi, step):
        noise = self.exp_noise.sample() / 10.
        if epi % MAX_EPOCH == 0:
            self.epsilon = EPSILON / (1. + (epi / EPI_DENOM))
        a_exp = noise * self.epsilon
        return torch.tensor(a_exp, dtype=torch.float, device=self.device)

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

from pid import PID


class InitialControl(object):
    def __init__(self, env, device):
        self.pid = PID(env, device)

    def controller(self, step, x, u):
        return self.pid.pid_ctrl(step, x)