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

        # hyperparameters
        self.max_episode = self.config.hyperparameters['max_episode']
        self.save_period = self.config.hyperparameters['save_period']

        self.path_data_history = []
        self.conv_stat_history = []
        self.reward_history = []

    def env_rollout(self):
        for epi in range(self.max_episode):
            epi_path_data = []
            epi_conv_stat = 0.
            epi_reward = 0.
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
                epi_conv_stat += nn_loss
                epi_reward += r

                if epi % self.save_period == 0:
                    epi_path_data.append([x, u_val, r, x2, y2, ref, derivs])

            self.postprocessing(epi_path_data, epi_conv_stat, epi_reward)

        return self.path_data_history, self.conv_stat_history, self.reward_history

    def save(self):
        pass

    def postprocessing(self, epi_path_data, epi_conv_stat, epi_reward):
        for path_data in zip(epi_path_data):
            x, u, r, x2, y2, ref, derivs = path_data

            x_record = np.reshape(x, [1, -1])
            y_record = np.reshape(y2, [1, -1])
            u_record = np.reshape(u, [1, -1])
            r_record = np.reshape(r, [1, -1])
            ref_record = np.reshape(ref, [1, -1])

            temp_data_history = np.concatenate([x_record, y_record, u_record, r_record, ref_record], 1).reshape([1, -1])

            self.path_data_history.extend(temp_data_history)

        self.conv_stat_history.extend(epi_conv_stat)
        self.reward_history.extend(epi_reward)

    def print_and_save_history(self, epi_num, prefix=None):
        if prefix is None:
            prefix = str('')

        np.set_printoptions(precision=4)
        print('| Episode ', '| Cost ', '| Conv ', '| Term cost ', '| Term conv ')
        print(epi_num,
              np.array2string(self.epi_stat_history[-1, :], formatter={'float_kind': lambda x: "    %.4f" % x}))

        np.savetxt('stat_history.txt', self.epi_stat_history, newline='\n')
        if epi_num % self.save_period == 0:
            np.savetxt('result/' + prefix + '_path_data_history.txt', self.epi_path_data_history, newline='\n')
            np.savetxt('result/' + prefix + '_term_data_history.txt', self.epi_term_data_history, newline='\n')


    def plot(self, epi_num):
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        fig = plt.figure(figsize=[20, 12])
        fig.subplots_adjust(hspace=.4, wspace=.5)
        label = [r'$C_{A}$', r'$C_{B}$', r'$T$', r'$T_{Q}$', r'$\frac{v}{V_{R}}$', r'$Q$',
                 r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$', r'$C_{B}$', r'$cost$']
        for j in range(len(label)):
            if label[j] in (r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$'):
                ax = fig.add_subplot(2, 6, j + 5)
            else:
                ax = fig.add_subplot(2, 6, j + 1)
            ax.plot(trajectory[1:, j + 1])
            if j in (1, 8):
                ax.plot(trajectory[1:, -1], ':g')
            plt.ylabel(label[j], size=8)
        plt.savefig('result/episode' + str(epi_num) + '.png')
        plt.show()