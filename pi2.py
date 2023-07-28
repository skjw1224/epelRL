import numpy as np
import utils
import torch

from replay_buffer import ReplayBuffer


class PI2(object):
    def __init__(self, config):
        self.config = config
        self.env = self.config.environment
        self.device = self.config.device

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.nT = self.env.nT

        # Hyperparameters
        self.init_ctrl_idx = self.config.hyperparameters['init_ctrl_idx']
        self.rbf_dim = self.config.hyperparameters['rbf_dim']
        self.rbf_type = self.config.hyperparameters['rbf_type']
        self.batch_epi = self.config.hyperparameters['batch_epi']

        self.explorer = self.config.algorithm['explorer']['function'](config)
        self.approximator = self.config.algorithm['approximator']['function']
        self.initial_ctrl = self.config.algorithm['controller']['initial_controller'](config)
        self.replay_buffer = ReplayBuffer(self.env, self.device, buffer_size=self.nT*self.batch_epi, batch_size=self.nT*self.batch_epi)

        # Actor network
        self.actor_net = self.approximator(self.s_dim, self.rbf_dim, self.rbf_type)
        self.theta = torch.randn([self.rbf_dim, self.a_dim])
        self.sigma = torch.ones([self.rbf_dim, self.a_dim])

    def ctrl(self, epi, step, s, a):
        if epi < self.init_ctrl_idx:
            a_nom = self.initial_ctrl.ctrl(epi, step, s, a)
            a_val = self.explorer.sample(epi, step, a_nom)
        else:
            a_val = self._choose_action(s)

        return a_val

    def _choose_action(self, s):
        pass

    def sampling(self, epi):
        for _ in range(self.batch_epi + 1):
            t, s, _, a = self.env.reset()
            for i in range(self.nT):
                a = self.ctrl(epi, i, s, a)
                t2, s2, _, r, is_term, _ = self.env.step(t, s, a)
                self.replay_buffer.add(*[s, a, r, s2, is_term])
                t, s = t2, s2

    def train(self):
        pass

    def choose_action(self, state, horizon):

        x_in_torch = utils.descale(state, self.xmin, self.xmax)  # torch
        x_in = x_in_torch.detach().numpy()   # torch -> numpy, descaled
        N = horizon

        """PI 하이퍼 파라미터"""
        epi_number = 100
        exploration_rate = np.array([[30, 300]])  # Heuristic 한 부분 (0으로 줘도 됨 = 빼도 됨)

        Up = np.zeros((epi_number, self.a_dim))     # epi_number x 2
        Below = np.zeros((epi_number, 1))           # epi_number x 1
        u_pi = np.zeros(self.a_dim)

        for k in range(epi_number):
            print("MC_episode: ", k)
            state_cost = np.zeros(N)
            initial_control_cost = np.zeros(N)
            total_state_cost = np.zeros(N)
            piu = np.zeros((N, self.a_dim))
            pig = np.zeros((N, self.a_dim, self.a_dim))
            w = np.zeros((N, self.a_dim))
            u_BS_sum, dum_z5, dum_z6 = self.Initial_ctrl(x_in)        # backsetpping control
            dum_z5 = np.squeeze(dum_z5)
            dum_z6 = np.squeeze(dum_z6)
            Gc = np.array([[-dum_z5, 0], [0, -dum_z6]])
            x = x_in

            for j in range(N):

                # Initial control
                u_BS, z5, z6 = self.Initial_ctrl(x)
                z5 = np.squeeze(z5)
                z6 = np.squeeze(z6)
                pig[j, :, :] = np.array([[-z5, 0], [0, -z6]])

                if j == 0:
                    piu[j] = exploration_rate * np.random.normal(0, 1, (1, self.a_dim))  # 1 x 2
                else:
                    piu[j] = np.zeros((1, self.a_dim))  # 1 x 2

                # Cost
                state_cost[j] = (x - self.Target_state) @ self.Q @ (x - self.Target_state).T  # rx(i)
                initial_control_cost[j] = u_BS.T @ self.R_bs @ u_BS  # rbs(i)
                total_state_cost[j] = self.Env.dt * (state_cost[j] + initial_control_cost[j])  # r(i)

                # Disturbance model
                u_MC = u_BS + pig[j, :, :] @ np.array([piu[j]]).T    # 2 x 1
                w[j] = np.random.normal(0, 1, (1, self.a_dim))  # 1 x 2
                Browian_motion = self.B(x)    # 7 x 2

                # Running Plant (Monte-Carlo search)
                u_MC = u_MC.T
                u_MC = u_MC.astype(np.float32)
                u_MC_torch = torch.from_numpy(u_MC)     # descaled
                u_MC_torch = utils.scale(u_MC_torch, self.umin, self.umax)  # torch, scaled
                x_torch = torch.from_numpy(x)
                x_torch = utils.scale(x_torch, self.xmin, self.xmax)    # torch, scaled

                xplus_torch, _, u_MC_torch, _, _, _ = self.Env.step(x_torch, u_MC_torch)

                xplus_torch = utils.descale(xplus_torch, self.xmin, self.xmax)
                xplus = xplus_torch.detach().numpy()    # 1 x 7
                xplus += w[j] @ Browian_motion.T * np.sqrt(self.Env.dt) # (1 x 2) @ (2 x 7) * scalar

                x = xplus

            Up[k, :] = np.exp(-np.sum(total_state_cost) / self.parameter_lambda) * (w[0] @ Browian_motion[5: ].T / np.sqrt(self.Env.dt) + piu[0] @ pig[0, :, :].T)
            Below[k] = np.exp(-np.sum(total_state_cost) / self.parameter_lambda)

        # Input derived by PI
        u_pi = sum(Up) / sum(Below) @ np.linalg.inv(Gc)   # 1 x 2

        # Integrating u_BS with u_PI
        u_BSPI = u_BS_sum.T + u_pi @ Gc.T
        u_BSPI = u_BSPI.astype(np.float32)
        u_BSPI_torch = torch.from_numpy(u_BSPI)
        u_BSPI_torch = utils.scale(u_BSPI_torch, self.umin, self.umax)

        return u_BSPI_torch   # 1 x 2

    def B(self, state):
        Browian_motion_coefficient = np.zeros((self.Env.s_dim, self.Env.a_dim)) # 7 x 2
        Browian_motion_coefficient[5, 0] = 3 * state[0][5]/200 + 0.3
        Browian_motion_coefficient[6, 1] = 3 * state[0][6]/80 + 3000
        return Browian_motion_coefficient

    def B_torch(self, state):
        Browian_motion_coefficient = torch.zeros((self.Env.s_dim, self.Env.a_dim)) # 7 x 2
        Browian_motion_coefficient[5, 0] = 3 * state[0][5]/200 + 0.3
        Browian_motion_coefficient[6, 1] = 3 * state[0][6]/80 + 3000
        return Browian_motion_coefficient
