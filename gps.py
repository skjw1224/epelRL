import numpy as np
import scipy as sp
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal


class TrajectoryInfo(object):
    def __init__(self, params):
        self.s_dim = params['s_dim']
        self.a_dim = params['a_dim']
        self.nT = params['nT']
        self.num_samples = params['num_samples']

        self.init_state = None
        self.sample_lst = np.zeros([self.num_samples, self.nT, self.s_dim + self.a_dim])

        self.local_K_lst = torch.ones([self.nT, self.a_dim, self.s_dim])
        self.local_k_lst = torch.ones([self.nT, self.a_dim, 1])
        self.local_C_lst = torch.eye(self.a_dim).expand(self.nT, self.a_dim, self.a_dim)

        self.global_K_lst = torch.ones([self.nT, self.a_dim, self.s_dim])
        self.global_k_lst = torch.ones([self.nT, self.a_dim, 1])
        self.global_C_lst = torch.eye(self.a_dim).expand(self.nT, self.a_dim, self.a_dim)

        self.policy_prior = []

        self.policy_variance = torch.eye(self.a_dim)

    def local_policy(self, s, t):
        K = self.local_K_lst[t]
        k = self.local_k_lst[t]
        C = self.local_C_lst[t]
        mu = K @ s + k
        sigma = C

        return mu, sigma

    def linearized_global_policy(self, s, t):
        K = self.global_K_lst[t]
        k = self.global_k_lst[t]
        C = self.global_C_lst[t]
        mu = K @ s + k
        sigma = C

        return mu, sigma


class PolicyPrior(object):
    def __init__(self, params):
        self.N = None
        self.D = params['s_dim'] + params['a_dim']
        self.K = params['num_clusters']
        self.max_iter = params['max_iter']
        self.sa_lst = None

    def update(self):
        # Run EM algorithm to update GMM
        self._initialization()
        loss = self._negative_log_likelihood()
        for _ in range(self.max_iter):
            self._e_step()
            self._m_step()
            loss = self._negative_log_likelihood()

    def _initialization(self):
        self.N = self.sa_lst.shape[0]
        self.responsibility = np.zeros((self.N, self.K))
        self.mu = np.random.rand(self.K, self.D)
        self.sigma = np.array([np.diag(np.random.rand(self.D)) for _ in range(self.K)])
        self.log_cluster_weights = np.log(1 / self.K) * np.ones((self.K, 1))

    def _e_step(self):
        log_responsibility = np.zeros((self.N, self.K))
        log_responsibility += -0.5 * np.ones((self.N, self.K)) * self.D * np.log(2*np.pi)
        for k in range(self.K):
            mu, sigma = self.mu[k], self.sigma[k]

            L = sp.linalg.cholesky(sigma, lower=True)
            log_responsibility[:, k] -= np.sum(np.log(np.diag(L)))

            diff = (self.sa_lst - mu).T
            soln = sp.linalg.solve_triangular(L, diff, lower=True)
            log_responsibility[:, k] -= 0.5 * np.sum(soln ** 2, axis=0)

        log_responsibility += self.log_cluster_weights.T
        log_responsibility -= np.repeat(np.log(np.sum(np.exp(log_responsibility), axis=1)).reshape(-1, 1), 3, axis=1)
        self.responsibility = np.exp(log_responsibility)

    def _m_step(self):
        Nk = np.sum(self.responsibility, axis=0).reshape(self.K, 1)
        self.mu = np.matmul(self.responsibility.T, self.sa_lst) / Nk
        for k in range(self.K):
            deviation = self.sa_lst - self.mu[k]
            self.sigma[k] = np.linalg.multi_dot([deviation.T, np.diag(self.responsibility[:, k]), deviation]) / Nk[k]
            self.sigma[k] = 0.5 * (self.sigma[k] + self.sigma[k].T) + 1e-7 * np.eye(self.D)
        self.log_cluster_weights = np.log(Nk / self.N)

    def _negative_log_likelihood(self):
        likelihood = np.zeros((self.N, self.K))
        likelihood += -0.5 * np.ones((self.N, self.K)) * self.D * np.log(2 * np.pi)
        for k in range(self.K):
            mu, sigma = self.mu[k], self.sigma[k]

            L = sp.linalg.cholesky(sigma, lower=True)
            likelihood[:, k] -= np.sum(np.log(np.diag(L)))

            diff = (self.sa_lst - mu).T
            soln = sp.linalg.solve_triangular(L, diff, lower=True)
            likelihood[:, k] -= 0.5 * np.sum(soln ** 2, axis=0)

        likelihood += self.log_cluster_weights.T
        likelihood = np.sum(np.exp(likelihood), axis=1)
        negative_log_likelihood = - np.sum(np.log(likelihood))

        return negative_log_likelihood

    def inference(self, sa_samples):
        N = sa_samples.shape[0]
        log_responsibility = np.zeros((N, self.K))
        log_responsibility += -0.5 * np.ones((N, self.K)) * self.D * np.log(2*np.pi)
        for k in range(self.K):
            mu, sigma = self.mu[k], self.sigma[k]

            L = sp.linalg.cholesky(sigma, lower=True)
            log_responsibility[:, k] -= np.sum(np.log(np.diag(L)))

            diff = (sa_samples - mu).T
            soln = sp.linalg.solve_triangular(L, diff, lower=True)
            log_responsibility[:, k] -= 0.5 * np.sum(soln ** 2, axis=0)

        log_responsibility += self.log_cluster_weights.T
        log_responsibility -= np.repeat(np.log(np.sum(np.exp(log_responsibility), axis=1)).reshape(-1, 1), 3, axis=1)

        Nk = np.sum(np.exp(log_responsibility), axis=0).reshape(self.K, 1)
        cluster_weights = Nk / N

        mu0 = np.sum(self.mu * cluster_weights, axis=0)

        diff = self.mu - np.expand_dims(mu0, axis=0)
        diff_expand = np.expand_dims(self.mu, axis=1) * np.expand_dims(diff, axis=2)
        wts_expand = np.expand_dims(cluster_weights, axis=2)
        Phi = np.sum((self.sigma + diff_expand) * wts_expand, axis=0)

        m, n0 = 1, 1

        return mu0, Phi, m, n0


class GPS(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        # Hyperparameters
        self.h_nodes = self.config.hyperparameters['hidden_nodes']
        self.buffer_size = self.config.hyperparameters['buffer_size']
        self.minibatch_size = self.config.hyperparameters['minibatch_size']
        self.learning_rate = self.config.hyperparameters['learning_rate']
        self.adam_eps = self.config.hyperparameters['adam_eps']
        self.l2_reg = self.config.hyperparameters['l2_reg']

        self.num_init_states = self.config.hyperparameters['num_init_states']
        self.num_samples = self.config.hyperparameters['num_samples']
        self.sampling_policy = self.config.hyperparameters['sampling_policy']
        self.base_kl_eps = self.config.hyperparameters['base_kl_eps']
        self.eta = self.config.hyperparameters['eta']
        self.min_eta = self.config.hyperparameters['min_eta']
        self.max_eta = self.config.hyperparameters['max_eta']
        self.dgd_max_iter = self.config.hyperparameters['dgd_max_iter']
        self.num_clusters = self.config.hyperparameters['num_clusters']
        self.max_iter = self.config.hyperparameters['max_iter']
        self.max_samples = self.config.hyperparameters['max_samples']

        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.initial_ctrl = [self.config.algorithm['controller']['initial_controller'](config) for _ in range(self.num_init_states)]

        # Trajectory information
        traj_params = {
            's_dim': self.s_dim,
            'a_dim': self.a_dim,
            'nT': self.nT,
            'num_samples': self.num_samples
        }
        self.traj_lst = [TrajectoryInfo(traj_params) for _ in range(self.num_init_states)]

        # Policy prior
        prior_params = {
            's_dim': self.s_dim,
            'a_dim': self.a_dim,
            'nT': self.nT,
            'num_clusters': self.num_clusters,
            'max_iter': self.max_iter
        }
        self.policy_prior = [PolicyPrior(prior_params) for _ in range(self.num_init_states)]

        # Global policy
        self.approximator = self.config.algorithm['approximator']['function']
        self.actor_net = self.approximator(self.s_dim, self.a_dim, self.h_nodes).to(self.device)
        self.actor_net_opt = optim.Adam(self.actor_net.parameters(), lr=self.learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Derivatives
        self.dx_derivs = self.env.dx_derivs
        self.c_derivs = self.env.c_derivs
        self.cT_derivs = self.env.cT_derivs

        # Parameter uncertainty
        self.p_mu, self.p_sigma, self.p_eps = self.env.p_mu, self.env.p_sigma, self.env.p_eps

        # Set initial states
        self._select_initial_states()

        self.loss_lst = ['Loss']

    def sampling(self, epi):
        # Sample mutliple trajectories from each initial state
        for m in range(self.num_init_states):
            self._sampling_traj(m)

    def train(self, epi):
        # Global policy linearization
        for m in range(self.num_init_states):
            self.policy_linearization(m)

        # C-step
        for m in range(self.num_init_states):
            self.c_step(m)

        # S-step
        self.s_step()

    def _select_initial_states(self):
        for m in range(self.num_init_states):
            _, x0, _, _ = self.env.reset(random_init=True)
            self.traj_lst[m].init_state = x0

    def _sampling_traj(self, m):
        time = self.env.t0
        s = self.traj_lst[m].init_state
        for n in range(self.num_samples):
            for t in range(self.nT):
                a = self._sampling_action(m, s, t)
                time, s2, _, _, _, _ = self.env.step(time, s, a)
                self.traj_lst[m].sample_lst[n, t, :self.s_dim] = s.reshape(-1, )
                self.traj_lst[m].sample_lst[n, t, self.s_dim:] = a.reshape(-1, )
                s = s2

    def _sampling_action(self, m, s, t):
        if self.sampling_policy == 'on_policy':
            s = torch.from_numpy(s.T).float().to(self.device)
            mu = self.actor_net(s)
            sigma = self.traj_lst[m].policy_variance.to(self.device)
            a_distribution = MultivariateNormal(mu, sigma)
        elif self.sampling_policy == 'off_policy':
            s = torch.from_numpy(s).float()
            mu, sigma = self.traj_lst[m].local_policy(s, t)
            a_distribution = MultivariateNormal(mu.T, sigma)
        else:
            raise ValueError('Wrong sampling policy')

        action = a_distribution.sample()

        return action.T.cpu().detach().numpy()

    def policy_linearization(self, m):
        sa_samples = self._sa_sample(m)
        self._refitting_policy_prior(m, sa_samples)
        mu0, Phi, mm, n0 = self._normal_inverse_wishart_prior(m, sa_samples)
        self._conditioning(m, sa_samples, mu0, Phi, mm, n0)

    def _sa_sample(self, m):
        # Choose s and a samples from current policy
        s_samples = self.traj_lst[m].sample_lst[:, :, :self.s_dim]
        s_samples = torch.from_numpy(s_samples).float().to(self.device)  # N*T*s_dim
        a_samples = self.actor_net(s_samples)  # N*T*a_dim
        sa_samples = torch.cat([s_samples, a_samples], dim=2)  # N*T*(s_dim+a_dim)

        return sa_samples.detach().cpu().numpy()

    def _empirical_mean_cov(self, sa_samples):
        # Compute empirical mean and covariance
        sa_mean = np.mean(sa_samples, axis=0)
        deviation = sa_samples - sa_mean
        sa_cov = np.zeros((self.nT, self.s_dim+self.a_dim, self.s_dim+self.a_dim))
        for t in range(self.nT):
            dev_t = deviation[:, t, :].squeeze()
            sa_cov[t, :, :] = dev_t.T.dot(dev_t)

        return sa_mean, sa_cov

    def _refitting_policy_prior(self, m, sa_samples):
        # Update policy prior
        prior = self.policy_prior[m]
        sa_samples = np.reshape(sa_samples, [self.nT*self.num_samples, self.s_dim+self.a_dim])  # NT*(s_dim+a_dim)

        if prior.sa_lst is None:
            prior.sa_lst = sa_samples
        else:
            prior.sa_lst = np.concatenate([prior.sa_lst, sa_samples], axis=0)
            N = prior.sa_lst.shape[0]
            if N > self.max_samples:
                start = N - self.max_samples
                prior.sa_lst = prior.sa_lst[start:]

        prior.update()

    def _normal_inverse_wishart_prior(self, m, sa_samples):
        # Obtain Normal-inverse-Wishart prior
        prior = self.policy_prior[m]
        sa_samples = np.reshape(sa_samples, [self.nT * self.num_samples, self.s_dim + self.a_dim])  # NT*(s_dim+a_dim)
        mu0, Phi, mm, n0 = prior.inference(sa_samples)

        return mu0, Phi, mm, n0

    def _conditioning(self, m, sa_samples, mu0, Phi, mm, n0):
        # Fit policy linearization by conditioning
        N = sa_samples.shape[0] * sa_samples.shape[1]
        T = sa_samples.shape[1]

        sa_mean, sa_cov = self._empirical_mean_cov(sa_samples)

        mu = (mm * mu0 + n0 * sa_mean) / (mm + n0)
        sigma = Phi + N * sa_cov
        for t in range(T):
            sigma[t] += (N * mm) / (N + mm) * np.outer(sa_mean[t] - mu0, sa_mean[t] - mu0)
            sigma[t] = 0.5 * (sigma[t] + sigma[t].T)
        sigma /= (N + n0)

        for t in range(self.nT):
            mu_s = mu[t, :self.s_dim]
            mu_a = mu[t, self.s_dim:]
            sigma_ss = sigma[t, :self.s_dim, :self.s_dim]
            sigma_aa = sigma[t, self.s_dim:, self.s_dim:]
            sigma_sa = sigma[t, :self.s_dim, self.s_dim:]

            Kt_bar = np.linalg.solve(sigma_ss, sigma_sa).T
            kt_bar = (mu_a - np.matmul(Kt_bar, mu_s)).reshape(-1, 1)
            Ct_bar = sigma_aa - np.matmul(Kt_bar, sigma_sa)
            Ct_bar = 0.5 * (Ct_bar + Ct_bar.T)

            self.traj_lst[m].global_K_lst[t] = torch.from_numpy(Kt_bar).float()
            self.traj_lst[m].global_k_lst[t] = torch.from_numpy(kt_bar).float()
            self.traj_lst[m].global_C_lst[t] = torch.from_numpy(Ct_bar).float()

    def c_step(self, m):
        kl_eps = self.nT * self.base_kl_eps
        eta = self.eta
        min_eta = self.min_eta
        max_eta = self.max_eta

        for itr in range(self.dgd_max_iter):
            # Run backward and forward pass
            self._backward(m, eta)
            mu, sigma = self._forward(m)

            # Compute KL divergence constraint
            kl_div = self._trajectory_kl_div(m, mu, sigma)
            convergence = kl_div - kl_eps

            # Check convergence
            if abs(convergence) < 0.1 * kl_eps:
                break

            # Choose new eta
            if convergence < 0:
                max_eta = eta
                eta = max(np.sqrt(min_eta * max_eta), 0.1 * max_eta)
            else:
                min_eta = eta
                eta = min(np.sqrt(min_eta * max_eta), 10.0 * min_eta)

        if kl_div > kl_eps and abs(kl_div - kl_eps) > 0.1 * kl_eps:
            print('Final KL divergence after DGD convergence is too high.')

    def _backward(self, m, eta):
        sample_lst = self.traj_lst[m].sample_lst

        xT, uT = sample_lst[-1]
        _, lxT, lxxT = [_.full() for _ in self.cT_derivs(xT, self.p_mu, self.p_sigma, self.p_eps)]
        Vxx = lxxT
        Vx = lxT

        for t in reversed(range(self.nT)):
            x, u = sample_lst[t]
            _, fx, fu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
            _, lx, lu, lxx, lxu, luu = [_.full() for _ in self.c_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]

            # TODO: linearization of global policy must be done beforehand
            Kt_bar = self.traj_lst[m].global_K_lst[t]
            kt_bar = self.traj_lst[m].global_k_lst[t]
            Ct_bar = self.traj_lst[m].global_C_lst[t]
            U = sp.linalg.cholesky(Ct_bar)
            Ct_bar_inv = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, np.eye(len(U)), lower=True))

            eps = 1E-7
            lx = (lx + Kt_bar.T @ Ct_bar_inv @ Kt_bar * eta) / (eta + eps)
            lu = (lu - Ct_bar_inv @ kt_bar * eta) / (eta + eps)
            lxx = (lxx + Kt_bar.T @ Ct_bar_inv @ Kt_bar * eta) / (eta + eps)
            lxu = (lxu - Kt_bar.T @ Ct_bar_inv * eta) / (eta + eps)
            luu = (luu + Ct_bar_inv * eta) / (eta + eps)

            Qx = lx + fx.T @ Vx
            Qu = lu + fu.T @ Vx
            Qxx = lxx + fx.T @ Vxx @ fx
            Qxx = 0.5 * (Qxx + Qxx.T)
            Quu = luu + fu.T @ Vxx @ fu
            Quu = 0.5 * (Quu + Quu.T)
            Qxu = lxu + fu.T @ Vxx @ fx

            try:
                U = sp.linalg.cholesky(Quu)
                Quu_inv = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, np.eye(len(U)), lower=True))
            except np.linalg.LinAlgError:
                Quu_inv = np.linalg.inv(Quu)

            kt = np.clip(- Quu_inv @ Qu, -1, 1)
            Kt = - Quu_inv @ Qxu.T
            Ct = Quu_inv
            self.traj_lst.local_K_lst.append(Kt)
            self.traj_lst.local_k_lst.append(kt)
            self.traj_lst.local_C_lst.append(Ct)

            Vx = Qx + Kt.T @ Quu @ kt + Kt.T @ Qu + Qxu @ kt
            Vxx = Qxx + Kt.T @ Quu @ Kt + Kt.T @ Qxu.T + Qxu @ Kt
            Vxx = 0.5 * (Vxx + Vxx.T)

        self.traj_lst.local_K_lst.reverse()
        self.traj_lst.local_k_lst.reverse()
        self.traj_lst.local_C_lst.reverse()

    def _forward(self, m):
        idx_x = slice(self.x_dim)

        mu = np.zeros((self.nT, self.x_dim + self.u_dim))
        sigma = np.zeros((self.nT, self.x_dim + self.u_dim, self.x_dim + self.u_dim))

        mu[0, idx_x] = self.traj_lst[m].x0mu
        sigma[0, idx_x, idx_x] = self.traj_lst[m].x0sigma

        for t in range(self.nT):
            K = self.traj_lst[m].local_K_lst[t]
            k = self.traj_lst[m].local_k_lst[t]
            C = self.traj_lst[m].local_C_lst[t]

            mu[t, :] = np.hstack([
                mu[t, idx_x],
                K.dot(mu[t, idx_x]) + k
            ])
            sigma[t, :, :] = np.vstack([
                np.hstack([
                    sigma[t, idx_x, idx_x],
                    sigma[t, idx_x, idx_x].dot(K.T)
                ]),
                np.hstack([
                    K.dot(sigma[t, idx_x, idx_x]),
                    K[t, :, :].dot(sigma[t, idx_x, idx_x]).dot(K.T) + C
                ])
            ])

            if t < self.nT - 1:
                sigma[t+1, idx_x, idx_x] = \
                        Fm[t, :, :].dot(sigma[t, :, :]).dot(Fm[t, :, :].T)
                mu[t+1, idx_x] = Fm[t, :, :].dot(mu[t, :]) + fv[t, :]

        return mu, sigma

    def _trajectory_kl_div(self, m, mu, sigma):
        kl_div = 0.
        for t in range(self.nT):
            K_local = self.traj_lst[m].local_K_lst[t]
            k_local = self.traj_lst[m].local_k_lst[t]
            C_local = self.traj_lst[m].local_C_lst[t]
            K_global = self.traj_lst[m].global_K_lst[t]
            k_global = self.traj_lst[m].global_k_lst[t]
            C_global = self.traj_lst[m].global_C_lst[t]

            U_local = sp.linalg.cholesky(C_local)
            U_global = sp.linalg.cholesky(C_global)

            log_det_local = 2 * np.sum(np.log(np.diag(U_local)))
            log_det_global = 2 * np.sum(np.log(np.diag(U_global)))
            precision_local = sp.linalg.solve_triangular(
                U_local, sp.linalg.solve_traingular(U_local.T, np.eye(len(U_local)), lower=True)
            )
            precision_global = sp.linalg.solve_triangular(
                U_global, sp.linalg.solve_traingular(U_global.T, np.eye(len(U_global)), lower=True)
            )

            M_global = np.r_[
                np.c_[K_global.T.dot(precision_global).dot(K_global), -K_global.T.dot(precision_global)],
                np.c_[-precision_global.dot(K_global), precision_global]
            ]
            M_local = np.r_[
                np.c_[K_local.T.dot(precision_local).dot(K_local), -K_local.T.dot(precision_local)],
                np.c_[-precision_local.dot(K_local), precision_local]
            ]
            v_global = np.r_[K_global.T.dot(precision_global).dot(k_global),
                           -precision_global.dot(k_global)]
            v_local = np.r_[K_local.T.dot(precision_local).dot(k_local), -precision_local.dot(k_local)]
            c_global = 0.5 * k_global.T.dot(precision_global).dot(k_global)
            c_local = 0.5 * k_local.T.dot(precision_local).dot(k_local)

            mu_t = mu[t, :]
            sigma_t = sigma[t, :, :]
            kl_div_t = max(
                0,
                -0.5 * mu_t.T.dot(M_local - M_global).dot(mu_t) -
                mu_t.T.dot(v_local - v_global) - c_local + c_global -
                0.5 * np.sum(sigma_t * (M_local-M_global)) - 0.5 * log_det_local +
                0.5 * log_det_global
            )

            kl_div += kl_div_t

        return kl_div

    def s_step(self):
        pass

