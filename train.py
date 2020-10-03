import numpy as np

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


    def env_rollout(self):
        for epi in range(self.config.hyperparameters['max_episode']):
            t0, x0, y0 = self.env.reset()
            trajectory = np.zeros([1, self.s_dim + self.o_dim + self.a_dim + 2])  # s + o + a + r + ref

            for i in range(self.nT):
                u = self.path_schedule(epi, i, x, u)
                # u_idx, u = self.dqn_controller.ctrl(epi, i, x, u_idx, r, x2, is_term)

                # u = trpo_controller.ctrl(epi, i, x, u, r, x2, is_term, derivs)
                # u = a2c_controller.ctrl(epi, i, x, u)
                # u = PoWER_controller.ctrl(epi, i, x, u)

                t2, x2, y2, u, r, is_term, derivs = self.env.step(t, x, u)
                # print("ode time:", time.time() - start_time)
                # a2c_controller.add_experience(x, u, r, x2, is_term)
                # PoWER_controller.add_experience(x, u, r, x2, is_term)
                # gdhp_controller.add_experience(x, u, r, x2, is_term, derivs)

                self.controller.add_experience(x, u, r, x2, is_term)
                self.controller.train()

                self.epi_path_data.append([x, x2, u])
                self.epi_path_misc_data.append([r, y2])

                x_record = np.reshape(x, [1, -1])
                u_record = np.reshape(u, [1, -1])
                y_record = np.reshape(y2, [1, -1])
                r_record = np.reshape(r, [1, -1])
                ref_record = np.reshape(env.scale(env.ref_traj(), env.ymin, env.ymax), [1, -1])
                step_data = np.concatenate([x_record, u_record, y_record, r_record, ref_record], axis=1)
                trajectory = np.concatenate([trajectory, step_data], axis=0)

                # Proceed loop
                t, x = t2, x2

            # Boundary rollout
            tT, xT, yT, uT, rT, derivs = self.env.step(t, x, u)

            self.epi_term_data.append([xT])
            self.epi_term_misc_data.append([rT, yT])

    def path_schedule(self, epi, i, x, u, epi_path_solution, prev_epi_path_data, training):
        if epi_path_solution is None:  # Initial control
            u = self.env.initial_control(i, x)

        else:  # Exists closed-loop control solution
            _, _, path_gain_epi, u_epi, l_epi, hypparam_epi, _, _ = epi_path_solution
            if training == True or prev_epi_path_data == None:
                # Training phase
                # x: simulated --> closed-loop solution already computed in the forward-sweep
                u = u_epi[i]
                l = l_epi[i]
            else:
                # Test phase
                # x: plant-data --> closed-loop solution need to be computed here
                x_nom = prev_epi_path_data[i][0]
                l, Kx, Ks, Kn = path_gain_epi[i]  # [lu, ll]; [Kux, Klx]; [Kus, Kls]; [Kun, Kln]
                delx = x - x_nom


                u = u_epi[i] + delu

        return u