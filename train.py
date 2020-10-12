import numpy as np
import utils

class Train(object):
    def __init__(self, config):
        self.config = config
        self.algorithm = self.config.algorithm
        self.controller = self.algorithm['controller']
        self.env = self.config.environment

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.o_dim = self.env.o_dim

        self.t0 = self.env.t0  # ex) 0
        self.tT = self.env.tT  # ex) 2
        self.nT = self.env.nT  # ex) dt:0.005 nT = 401

        self.epi_path_data, self.epi_term_data = [], []
        self.epi_conv_stat = 0.
        self.epi_reward = 0.

    def env_rollout(self):
        for epi in range(self.config.hyperparameters['max_episode']):
            # Initialize
            t, x, y = self.env.reset()
            for i in range(self.nT + 1):
                u = self.controller.ctrl(epi, i, x, u)

                if self.controller['action_type'] == 'discrete':
                    u_val = utils.action_idx2mesh(u, *self.config.algorithm['action_mesh'])
                else:
                    u_val = u

                # u = trpo_controller.ctrl(epi, i, x, u, r, x2, is_term, derivs)
                # u = PoWER_controller.ctrl(epi, i, x, u)

                t2, x2, y2, r, is_term, derivs = self.env.step(t, x, u_val)
                # print("ode time:", time.time() - start_time)
                # PoWER_controller.add_experience(x, u, r, x2, is_term)

                ref = np.reshape(self.env.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax), [1, -1])

                if self.controller['model_using'] == 'model_based':
                    self.controller.add_experience(x, u, r, x2, is_term, derivs)
                else:
                    self.controller.add_experience(x, u, r, x2, is_term)

                nn_loss = self.controller.train(i)

                # Proceed loop
                t, x = t2, x2

                # Save data
                self.epi_conv_stat += nn_loss
                self.epi_reward += r
                self.epi_path_data.append([x, u_val, r, x2, y2, ref, derivs])

        return self.epi_path_data, self.epi_conv_stat, self.epi_reward