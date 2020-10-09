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
        self.epi_path_misc_data, self.epi_term_misc_data = [], []


    def env_rollout(self):
        for epi in range(self.config.hyperparameters['max_episode']):
            t, x, y = self.env.reset()

            for i in range(self.nT):
                u = self.controller.ctrl(epi, i, x, u)

                if self.controller['action_type'] == 'discrete':
                    u_val = utils.action_idx2mesh(u, *self.config.algorithm['action_mesh'])
                else:
                    u_val = u

                # u = trpo_controller.ctrl(epi, i, x, u, r, x2, is_term, derivs)
                # u = a2c_controller.ctrl(epi, i, x, u)
                # u = PoWER_controller.ctrl(epi, i, x, u)

                t2, x2, y2, r, is_term, derivs = self.env.step(t, x, u_val)
                # print("ode time:", time.time() - start_time)
                # a2c_controller.add_experience(x, u, r, x2, is_term)
                # PoWER_controller.add_experience(x, u, r, x2, is_term)
                # gdhp_controller.add_experience(x, u, r, x2, is_term, derivs)

                ref = np.reshape(self.env.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax), [1, -1])

                self.controller.add_experience(x, u, r, x2, is_term)
                self.controller.train(i)

                self.epi_path_data.append([x, x2, u_val])
                self.epi_path_misc_data.append([r, y2, ref, derivs])

                # Proceed loop
                t, x = t2, x2

            # Boundary rollout
            tT, xT, yT, rT, derivs = self.env.step(t, x, u_val)

            self.epi_term_data.append([xT])
            self.epi_term_misc_data.append([rT, yT])

        epi_data = [self.epi_path_data, self.epi_path_misc_data]
        epi_misc_data = [self.epi_term_data, self.epi_term_misc_data]

        return epi_data, epi_misc_data