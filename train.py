import numpy as np
import matplotlib.pyplot as plt
import utils
import pickle
import os

class Train(object):
    def __init__(self, config):
        self.config = config
        self.algorithm = self.config.algorithm
        self.controller = self.algorithm['controller'](config)
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
        self.result_save_path = self.config.result_save_path
        self.plot_snapshot = self.config.hyperparameters['plot_snapshot']

    def env_rollout(self):
        for epi in range(self.max_episode):
            epi_path_data = []
            epi_conv_stat = 0.
            epi_reward = 0.
            # Initialize
            t, x, y = self.env.reset()
            u = None
            for i in range(self.nT + 1):
                u = self.controller.ctrl(epi, i, x, u)

                if self.algorithm['action_type'] == 'discrete':
                    u_val = utils.action_idx2mesh(u, *self.algorithm['action_mesh_idx'])
                else:
                    u_val = u

                t2, x2, y2, r, is_term, derivs = self.env.step(t, x, u_val)

                ref = np.reshape(self.env.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax), [1, -1])

                if self.algorithm['model_requirement'] == 'model_based':
                    self.controller.add_experience(x, u, r, x2, is_term, derivs)
                else:
                    self.controller.add_experience(x, u, r, x2, is_term)

                nn_loss = self.controller.train(step=i)

                # Proceed loop
                t, x = t2, x2

                # Save data
                epi_conv_stat += nn_loss
                epi_reward += r.item()

                if epi % self.save_period == 0:
                    epi_path_data.append([x, u_val, r, x2, y2, ref])

            traj_data_history, stat_history = self.postprocessing(epi_path_data, epi_conv_stat, epi_reward)

            self.print_stats(stat_history, epi_num=epi)
        self.save(traj_data_history, stat_history)

    def postprocessing(self, epi_path_data, epi_conv_stat, epi_reward):
        traj_data_history = []
        stat_history = []
        for path_data in epi_path_data:
            x, u, r, x2, y2, ref = path_data

            x_record = np.reshape(x, [1, -1])
            y_record = np.reshape(y2, [1, -1])
            u_record = np.reshape(u, [1, -1])
            r_record = np.reshape(r, [1, -1])
            ref_record = np.reshape(ref, [1, -1])

            temp_data_history = np.concatenate([x_record, y_record, u_record, r_record, ref_record], 1).reshape([1, -1])

            traj_data_history.append(temp_data_history)

        stat_history.append(np.array([epi_conv_stat, epi_reward]))

        return traj_data_history, stat_history

    def print_stats(self, stat_history, epi_num):
        np.set_printoptions(precision=4)
        print('| Episode ', '| Cost ', '| Conv ')
        print(epi_num,
              np.array2string(stat_history[-1], formatter={'float_kind': lambda x: "    %.4f" % x})[1:-1])

    def save(self, traj_data_history, stat_history):
        file_name = self.result_save_path + 'final_solution.pkl'
        if not os.path.exists(file_name):
            with open(self.result_save_path + 'final_solution.pkl', 'wb') as fw:
                solution = [traj_data_history, stat_history]
                pickle.dump(solution, fw)

    def plot(self):
        with open('final_solution.pkl', 'rb') as fr:
            solution = pickle.load(fr)
        traj_data_history, stat_history = solution

        traj_data_history = np.squeeze(np.array(traj_data_history), axis=1)

        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        fig = plt.figure(figsize=[20, 12])
        fig.subplots_adjust(hspace=.4, wspace=.5)
        label = [r'$C_{A}$', r'$C_{B}$', r'$T$', r'$T_{Q}$', r'$\frac{v}{V_{R}}$', r'$Q$',
                 r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$', r'$C_{B}$', r'$cost$']

        for epi_num in self.plot_snapshot:
            for j in range(len(label)):
                if label[j] in (r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$'):
                    ax = fig.add_subplot(2, 6, j + 5)
                else:
                    ax = fig.add_subplot(2, 6, j + 1)
                ax.save(traj_data_history[1:, j + 1])
                if j in (1, 8):
                    ax.save(traj_data_history[1:, -1], ':g')
                plt.ylabel(label[j], size=8)
        plt.savefig(self.result_save_path + str(epi_num) + 'plot.png')
        plt.show()