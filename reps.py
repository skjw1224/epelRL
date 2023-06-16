import torch
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from replay_buffer import ReplayBuffer


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
        self.rbf_dim = self.config.hyperparameters['rbf_dim']
        self.rbf_type = self.config.hyperparameters['rbf_type']
        self.batch_epi = self.config.hyperparameters['batch_epi']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT*self.batch_epi, batch_size=self.nT*self.batch_epi)

        self.rbf = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type)

        self.bounds = [(0, None)]*self.nT+[(None, None)]*(self.nT*self.rbf_dim)
        self.eta = 1 + torch.rand([self.nT], dtype=torch.float, device=self.device)
        self.theta = torch.rand([self.nT*self.rbf_dim], dtype=torch.float, device=self.device)
        self.st = torch.zeros([self.nT, self.a_dim], dtype=torch.float, device=self.device)
        self.St = torch.zeros([self.nT, self.a_dim, self.s_dim], dtype=torch.float, device=self.device)
        self.stdt = 20*torch.eye(self.a_dim, dtype=torch.float, device=self.device).expand_as(torch.zeros([self.nT, self.a_dim, self.a_dim]))

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

    def train(self):
        self.policy_update()
        print('policy improved')
        self.epoch = 0  # reset the number of single paths after training
        self.replay_buffer.clear()

    def policy_update(self):
        # eta = self.eta.cpu().detach().numpy()
        # theta = self.theta.cpu().detach().numpy().reshape(self.rbf_dim*self.end_t)
        # x0 = np.concatenate([eta,theta])
        x0 = torch.cat([self.eta, self.theta]).cpu()
        sol = fmin_l_bfgs_b(self.solve_dualfunc, x0, bounds=self.bounds)
        self.eta = torch.tensor(sol[0][:self.nT], dtype=torch.float, device=self.device)
        self.theta = torch.tensor(sol[0][self.nT:], dtype=torch.float, device=self.device)

        self.get_weightedML()

    def solve_dualfunc(self, x):
        x = x.reshape([self.nT, 1+self.rbf_dim])
        x = torch.tensor(x, dtype=torch.float64, device=self.device)
        eta = x[:,0]
        theta = x[:,1:]

        s_batch, a_batch, r_batch, s2_batch, term_batch = self.replay_buffer.sample_sequence()

        s_batch = s_batch.double()
        a_batch = a_batch.double()
        r_batch = r_batch.double()
        s2_batch = s2_batch.double()
        term_batch = term_batch.double()

        phi_batch = self.rbf(s_batch)
        phi2_batch = self.rbf(s2_batch)

        dual = torch.zeros([1], dtype=torch.float64, device=self.device)
        dual_deta = torch.zeros([self.nT], dtype=torch.float64, device=self.device)
        dual_dtheta = torch.zeros([self.nT*self.rbf_dim], dtype=torch.float64, device=self.device)

        log_vec_prev = torch.ones([self.batch_epi], dtype=torch.float64, device=self.device)

        for t in range(self.nT):
            # log_sum = torch.zeros([1], device=self.device)
            # pt = torch.zeros([self.batch_epi])
            # ptj_sum = torch.zeros([1])
            delta = torch.zeros([self.batch_epi], dtype=torch.float64, device=self.device)
            d_delta = torch.zeros([self.batch_epi, self.s_dim], dtype=torch.float64, device=self.device)
            d_delta_prev = torch.zeros([self.batch_epi, self.s_dim], dtype=torch.float64, device=self.device)
            r_batch_t = torch.zeros([self.batch_epi], dtype=torch.float64, device=self.device)
            phi = torch.zeros([self.batch_epi, self.s_dim], dtype=torch.float64, device=self.device)
            phi2_batch_t = torch.zeros([self.batch_epi, self.s_dim], dtype=torch.float64, device=self.device)
            a_batch_t = torch.zeros([self.batch_epi, self.a_dim], dtype=torch.float64, device=self.device)
            # term_batch_t = torch.zeros([self.batch_epi])

            for j in range(self.batch_epi):
                r_batch_t[j] = r_batch[t + j * self.nT]
                phi[j] = phi_batch[t + j * self.nT]
                phi2_batch_t[j] = phi2_batch[t + j * self.nT]
                a_batch_t[j] = a_batch[t + j * self.nT]
                # term_batch_t[j] = term_batch[t+j*self.nT]

                if term_batch[t + j * self.nT]:
                    delta[j] = r_batch[t + j * self.nT] - torch.matmul(phi_batch[t + j * self.nT], theta[t].T)
                else:
                    delta[j] = r_batch[t + j * self.nT] + torch.matmul(phi2_batch[t + j * self.nT], theta[t + 1].T) \
                               - torch.matmul(phi_batch[t + j * self.nT], theta[t].T)

                d_delta[j] = -phi[j]
                d_delta_prev[j] = phi[j]

            pt = torch.exp(delta / eta[t])
            pt /= torch.sum(pt)

            log_vec = pt * torch.exp(0.01 + delta / eta[t])
            log_sum = torch.sum(log_vec)
            dual = dual + eta[t] * torch.log(log_sum)

            log_vec_eta = log_vec * delta
            log_sum_eta = torch.sum(log_vec_eta)
            dual_deta[t] = torch.log(log_sum) - (1/eta[t])**2*log_sum_eta/log_sum

            log_sum_theta = torch.matmul(log_vec,d_delta)

            log_sum_prev = torch.sum(log_vec_prev)
            log_sum_theta_prev = torch.matmul(log_vec_prev,d_delta_prev)
            dual_dtheta[t*self.rbf_dim:(t+1)*self.rbf_dim] = log_sum_theta/log_sum + log_sum_theta_prev/log_sum_prev

            log_vec_prev = log_vec

        dual = dual.cpu().detach().numpy()
        d_dual = torch.cat([dual_deta,dual_dtheta])
        d_dual = d_dual.cpu().detach().numpy()

        return dual, d_dual

    def get_weightedML(self):

        s_batch, a_batch, r_batch, s2_batch, term_batch, _, _, _, _ = self.replay_memory.sample_sequence()

        for t in range(self.nT):
            # pt = torch.zeros([self.batch_epi])
            # ptj_sum = torch.zeros([1])
            delta = torch.zeros([self.batch_epi], dtype=torch.float, device=self.device)
            Xt = torch.ones([self.batch_epi,1+self.s_dim], dtype=torch.float, device=self.device)
            Ut = torch.zeros([self.batch_epi,self.a_dim], dtype=torch.float, device=self.device)

            r_batch_t = torch.zeros([self.batch_epi], dtype=torch.float, device=self.device)
            s_batch_t = torch.zeros([self.batch_epi,self.s_dim], dtype=torch.float, device=self.device)
            s2_batch_t = torch.zeros([self.batch_epi,self.s_dim], dtype=torch.float, device=self.device)
            a_batch_t = torch.zeros([self.batch_epi,self.a_dim], dtype=torch.float, device=self.device)
            # term_batch_t = torch.zeros([self.batch_epi])

            for j in range(self.batch_epi):
                r_batch_t[j] = r_batch[t+j*self.nT]
                s_batch_t[j] = s_batch[t+j*self.nT]
                s2_batch_t[j] = s2_batch[t+j*self.nT]
                a_batch_t[j] = a_batch[t+j*self.nT]
                # term_batch_t[j] = term_batch[t+j*self.nT]

                if term_batch[t+j*self.nT]:
                    delta[j] = r_batch[t+j*self.nT] - torch.matmul(s_batch[t+j*self.nT],self.theta[t*self.rbf_dim:(t+1)*self.rbf_dim].T)
                else:
                    delta[j] = r_batch[t+j*self.nT] + torch.matmul(s2_batch[t+j*self.nT],self.theta[t*self.rbf_dim:(t+1)*self.rbf_dim].T) \
                            - torch.matmul(s_batch[t+j*self.nT],self.theta[t*self.rbf_dim:(t+1)*self.rbf_dim].T)

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

            mut = torch.zeros([self.batch_epi,self.a_dim], dtype=torch.float, device=self.device )
            for k in range(self.batch_epi):
                mut[k] = self.st[t] + torch.matmul(self.St[t],s_batch_t[k])

            er = mut - a_batch_t
            self.stdt[t] = torch.mm(torch.mm(er.T, Dt), er)/torch.sum(pt)

        # del s_batch, a_batch, r_batch, s2_batch, term_batch


