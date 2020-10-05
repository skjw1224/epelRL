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

                t2, x2, y2, u, r, is_term, derivs = self.env.step(t, x, u_val)
                # print("ode time:", time.time() - start_time)
                # a2c_controller.add_experience(x, u, r, x2, is_term)
                # PoWER_controller.add_experience(x, u, r, x2, is_term)
                # gdhp_controller.add_experience(x, u, r, x2, is_term, derivs)

                ref = np.reshape(self.env.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax), [1, -1])

                self.controller.add_experience(x, u, r, x2, is_term)
                self.controller.train()

                self.epi_path_data.append([x, x2, u])
                self.epi_path_misc_data.append([r, y2, ref, derivs])

                # Proceed loop
                t, x = t2, x2

            # Boundary rollout
            tT, xT, yT, uT, rT, derivs = self.env.step(t, x, u)

            self.epi_term_data.append([xT])
            self.epi_term_misc_data.append([rT, yT])

        epi_data = [self.epi_path_data, self.epi_path_misc_data]
        epi_misc_data = [self.epi_term_data, self.epi_term_misc_data]

        return epi_data, epi_misc_data

    def path_schedule(self, epi, i, x, u, epi_path_solution, prev_epi_path_data, training):
        if epi_path_solution is None:  # Initial control
            u = self.controller.initial_control(epi, i, x, u)

        else:  # Exists closed-loop control solution
            u = self.controller.ctrl(epi, i, x, u)

            # _, _, path_gain_epi, u_epi, l_epi, hypparam_epi, _, _ = epi_path_solution
            # if training == True or prev_epi_path_data == None:
            #     # Training phase
            #     # x: simulated --> closed-loop solution already computed in the forward-sweep
            #     u = u_epi[i]
            #     l = l_epi[i]
            # else:
            #     # Test phase
            #     # x: plant-data --> closed-loop solution need to be computed here
            #     x_nom = prev_epi_path_data[i][0]
            #     l, Kx, Ks, Kn = path_gain_epi[i]  # [lu, ll]; [Kux, Klx]; [Kus, Kls]; [Kun, Kln]
            #     delx = x - x_nom
            #
            #
            #     u = u_epi[i] + delu

        return u