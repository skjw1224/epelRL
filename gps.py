import numpy as np
import scipy as sp
import torch
import torch.optim as optim
from torch.distributions import MultivariateNormal

from replay_buffer import ReplayBuffer


class BundleType(object):
    """
    This class bundles many fields, similar to a record or a mutable namedtuple.
    """
    def __init__(self, variables):
        for var, val in variables.items():
            object.__setattr__(self, var, val)

    # Freeze fields so new ones cannot be set.
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)


class Trajectory(BundleType):
    def __init__(self):
        variables = {
            'init_state': None,
            'sample_lst': [],
            'cost': [],
            'local_K_lst': [],
            'local_k_lst': [],
            'local_C_lst': [],
            'global_K_lst': [],
            'global_k_lst': [],
            'global_C_lst': [],
            'policy_prior': None,
        }
        BundleType.__init__(self, variables)

    def local_policy(self, x, time):
        K = self.local_K_lst[time]
        k = self.local_k_lst[time]
        C = self.local_C_lst[time]
        mu = K @ x + k
        sigma = C

        return mu, sigma

    def global_policy_variance(self):
        # TODO: computation of global policy variance
        sigma = None

        return sigma

    def linearized_global_policy(self, x, time):
        K = self.global_K_lst[time]
        k = self.global_k_lst[time]
        C = self.global_C_lst[time]
        mu = K @ x + k
        sigma = C

        return mu, sigma

# TODO: modify variable names
class PolicyPrior(object):
    def __init__(self, config):
        self.config = config
        self.xu_lst = None
        self.K = self.config.hyperparameters['num_clusters']

    def update(self):
        self._initialization()
        _ = self._negative_log_likelihood()
        for iter in range(self.max_iter):
            self._e_step()
            self._m_step()
            loss = self._negative_log_likelihood()

    def _initialization(self):
        self.responsibilities = np.zeros((self.N, self.K))
        self.mu = np.random.rand(self.D, self.K)
        self.sigma = np.array([np.diag(np.random.rand(self.D)) for _ in range(self.K)])
        self.pi = np.ones((self.K, 1)) / self.K

    def _negative_log_likelihood(self):
        temp = np.zeros((self.N, self.K))
        for k in range(self.K):
            temp[:, k] = sp.stats.multivariate_normal.pdf(self.xu_lst, self.mu[k], self.sigma[k])
        temp *= self.pi.T
        negative_log_likelihood = - np.sum(np.log10(np.sum(temp, axis=0)))

        return negative_log_likelihood

    def _e_step(self):
        temp = np.zeros((self.N, self.K))
        for k in range(self.K):
            temp[:, k] = sp.stats.multivariate_normal.pdf(self.xu_lst, self.mu[k], self.sigma[k])
        temp *= self.pi.T
        self.responsibilities = temp / np.sum(temp, axis=1).reshape(self.N, -1)

    def _m_step(self):
        Nk = np.sum(self.responsibilities, axis=0).reshape(self.K, -1)
        self.mu = np.matmul(self.responsibilities.T, self.xu_lst) / Nk
        for k in range(self.K):
            deviation = self.xu_lst - self.mu[k]
            self.sigma[k] = np.linalg.multi_dot([deviation.T, np.diag(self.responsibilities[:, k]), deviation]) / Nk[k]
        self.pi = Nk / self.N


class GPS(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.x_dim = self.env.s_dim
        self.u_dim = self.env.a_dim
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

        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.initial_ctrl = [self.config.algorithm['controller']['initial_controller'](config) for _ in range(self.num_init_states)]

        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.buffer_size, batch_size=self.minibatch_size)

        # Trajectory information
        self.traj_lst = [Trajectory() for _ in range(self.num_init_states)]

        # Policy prior
        self.policy_prior = [PolicyPrior(config) for _ in range(self.num_init_states)]

        # Global policy
        self.approximator = self.config.algorithm['approximator']['function']
        self.actor_net = self.approximator(self.x_dim, self.u_dim, self.h_nodes).to(self.device)
        self.actor_net_opt = optim.Adam(self.actor_net.parameters(), lr=self.learning_rate, eps=self.adam_eps, weight_decay=self.l2_reg)

        # Derivatives
        self.dx_derivs = self.env.dx_derivs
        self.c_derivs = self.env.c_derivs
        self.cT_derivs = self.env.cT_derivs

        # Parameter uncertainty
        self.p_mu, self.p_sigma, self.p_eps = self.env.p_mu, self.env.p_sigma, self.env.p_eps

        # Set initial states
        self._select_initial_states()

    def sampling(self, epi):
        # Sample mutliple trajectories from each initial states
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
        x = self.traj_lst[m].init_state
        for n in range(self.num_samples):
            sample = []
            for t in range(self.nT):
                u_distribution = self._action_distribution(m, x, t)
                u = u_distribution.sample()
                sample.append((x, u))
                time, x_next, _, _, _, _ = self.env.step(time, x, u)
                x = x_next
            self.traj_lst[m].sample_lst.append(sample)

    def _action_distribution(self, m, x, t):
        if self.sampling_policy == 'on_policy':
            mu = self.actor_net(x)
            sigma = self.traj_lst[m].global_policy_variance()
            u_distribution = MultivariateNormal(mu, sigma)
        elif self.sampling_policy == 'off_policy':
            mu, sigma = self.traj_lst[m].local_policy(x, t)
            u_distribution = MultivariateNormal(mu, sigma)
        else:
            print('Inappropriate sampling policy')
            u_distribution = None

        return u_distribution

    def policy_linearization(self, m):
        xu_sample = self._xu_sample(m)
        xu_mean, xu_cov = self._empirical_mean_cov(xu_sample)
        self._gmm_update(m, xu_sample)
        mu0, Phi, mm, n0 = self._normal_inverse_wishart_prior(m)
        self._conditioning(m, xu_mean, xu_cov, mu0, Phi, mm, n0)

    def _xu_sample(self, m):
        # Choose x and u samples
        sample_lst = self.traj_lst[m].sample_lst
        x_sample = np.zeros((self.num_samples, self.nT, self.x_dim))
        for n in range(self.num_samples):
            for t in range(self.nT):
                x_sample[n, t, :] = sample_lst[n][t][0]
        x_sample = torch.tensor(x_sample, dtype=torch.float32).to(self.device)
        u_sample = self.actor_net(x_sample)
        x_sample = x_sample.detach().cpu().numpy()  # N*T*x_dim
        u_sample = u_sample.detach().cpu().numpy()  # N*T*u_dim
        xu_sample = np.concatenate((x_sample, u_sample), axis=2)  # N*T*(x_dim+u_dim)

        return xu_sample

    def _empirical_mean_cov(self, xu_sample):
        # Compute empirical mean and covariance
        xu_mean = np.mean(xu_sample, axis=0)
        deviation = xu_sample - xu_mean
        xu_cov = np.zeros((self.nT, self.x_dim+self.u_dim, self.x_dim+self.u_dim))
        for t in range(self.nT):
            dev_t = deviation[:, t, :].squeeze()
            xu_cov[t, :, :] = dev_t.T.dot(dev_t)

        return xu_mean, xu_cov

    def _gmm_update(self, m, xu_sample):
        # Update policy prior
        prior = self.policy_prior[m]
        xu_sample = np.reshape(xu_sample, [self.nT*self.num_samples, self.x_dim+self.u_dim])  # NT*(x_dim+u_dim)

        if prior.xu_lst is None:
            prior.xu_lst = xu_sample
        else:
            prior.xu_lst = np.concatenate([prior.xu_lst, xu_sample])

        prior.update()

    def _normal_inverse_wishart_prior(self, m):
        # Obtain Normal-inverse-Wishart prior
        # TODO: modify mu0, Phi
        prior = self.policy_prior[m]

        mu0 = np.sum(prior.mu, axis=1)
        Phi = np.sum(prior.sigma, axis=2)
        mm = 1
        n0 = 1

        return mu0, Phi, mm, n0

    def _conditioning(self, m, xu_mean, xu_cov, mu0, Phi, mm, n0):
        # Fit policy linearization by conditioning
        # TODO: check the dimension of each matrix
        N = self.num_samples
        mu = (mm * mu0 + n0 * xu_mean) / (mm + n0)
        sigma = (Phi + N * xu_cov + (N * mm) / (N + mm) * np.outer(xu_mean - mu0, xu_mean - mu0)) / (N + n0)
        sigma = 0.5 * (sigma + sigma.T)

        for t in range(self.nT):
            Kt_bar = np.linalg.solve(sigma[:self.x_dim,:self.x_dim], sigma[:self.x_dim, self.x_dim:self.x_dim+self.u_dim]).T
            kt_bar = mu[self.x_dim:self.x_dim+self.u_dim] - Kt_bar.dot(mu[:self.x_dim])
            Ct_bar = sigma[self.x_dim:self.x_dim+self.u_dim, self.x_dim:self.x_dim+self.u_dim] - Kt_bar.dot(sigma[:self.x_dim, :self.x_dim]).dot(Kt_bar.T)
            Ct_bar = 0.5 * (Ct_bar + Ct_bar.T)

            self.traj_lst[m].global_K_lst.append(Kt_bar)
            self.traj_lst[m].global_k_lst.append(kt_bar)
            self.traj_lst[m].global_C_lst.append(Ct_bar)

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

