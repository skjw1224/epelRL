import numpy as np
import matplotlib.pyplot as plt
import utils
import pickle
import os

class Train(object):
    def __init__(self, config):
        self.config = config
        self.controller = self.config.algorithm['controller']['function'](config)
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

        self.traj_data_history = []
        self.stat_history = []

    def env_rollout(self):
        for epi in range(self.max_episode + 1):
            epi_path_data = []
            epi_conv_stat = 0.
            epi_reward = 0.
            # Initialize
            t, x, y, u = self.env.reset()
            for i in range(self.nT):
                u = self.controller.ctrl(epi, i, x, u)

                if self.config.algorithm['controller']['action_type'] == 'discrete':
                    u_val = utils.action_idx2mesh(u, *self.config.algorithm['controller']['action_mesh_idx'])
                else:
                    u_val = u

                t2, x2, y2, r, is_term, derivs = self.env.step(t, x, u_val)

                ref = self.env.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax).reshape([1, -1])

                if self.config.algorithm['controller']['model_requirement'] == 'model_based':
                    self.controller.add_experience(x, u, r, x2, is_term, derivs)
                else:
                    self.controller.add_experience(x, u, r, x2, is_term)

                nn_loss = self.controller.train(step=i)

                # Proceed loop
                t, x = t2, x2

                # Save data
                epi_reward += r.item()
                epi_conv_stat += nn_loss
                if epi % self.save_period == 0:
                    epi_path_data.append([x, u_val, r, x2, y2, ref])

            # TODO: if - convergence
            # END
            # 수렴한 에피소드 프린트해주기

            self.postprocessing(epi_path_data, epi_reward, epi_conv_stat)

            self.print_stats(self.stat_history, epi_num=epi)

        self.save(self.traj_data_history, self.stat_history)

    def postprocessing(self, epi_path_data, epi_reward, epi_conv_stat):
        for path_data in epi_path_data:
            x, u, r, x2, y2, ref = path_data

            x_record = self.env.descale(x, self.env.xmin, self.env.xmax).reshape([1, -1])
            y_record = self.env.descale(y2, self.env.ymin, self.env.ymax).reshape([1, -1])
            u_record = self.env.descale(u, self.env.umin, self.env.umax).reshape([1, -1])
            r_record = r.reshape([1, -1])
            ref_record = self.env.descale(ref, self.env.ymin, self.env.ymax).reshape([1, -1])

            temp_data_history = np.concatenate([x_record, y_record, u_record, r_record, ref_record], 1).reshape([1, -1])

            self.traj_data_history.append(temp_data_history)

        self.stat_history.append(np.array([epi_reward, epi_conv_stat]))

    def print_stats(self, stat_history, epi_num):
        np.set_printoptions(precision=4)
        print('| Episode ', '| Cost ', '| Conv ')
        print(epi_num,
              np.array2string(stat_history[-1], formatter={'float_kind': lambda x: "    %.4f" % x})[1:-1])

    def save(self, traj_data_history, stat_history):
        with open(self.result_save_path + 'final_solution.pkl', 'wb') as fw:
            solution = [traj_data_history, stat_history]
            pickle.dump(solution, fw)

    def plot(self):
        with open(self.result_save_path + 'final_solution.pkl', 'rb') as fr:
            solution = pickle.load(fr)
        traj_data_history, stat_history = solution

        num_ep = int(np.shape(traj_data_history)[0] / self.nT)
        traj_data_history = np.array(traj_data_history).reshape([num_ep, self.nT, -1])
        stat_history = np.array(stat_history)

        fig = plt.figure(0, figsize=[20, 12])
        fig.subplots_adjust(hspace=.4, wspace=.5)
        x_label = [r'$C_{A}$', r'$C_{B}$', r'$T_{R}$', r'$T_{C}$', r'$\frac{\dot{V}}{V_{R}}$', r'$\dot{Q}$']
        u_label = [r'$\frac{\Delta\dot{V}}{V_{R}}$', r'$\Delta\dot{Q}$', r'$C_{B}$']

        colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
        ax_list = []

        for e, epi_num in enumerate(self.plot_snapshot):
            epi_num = int(epi_num / self.save_period)
            tgrid = traj_data_history[epi_num, :, 0]

            for i in range(self.s_dim - 1):
                if e == 0:
                    ax = fig.add_subplot(2, 4, i + 1)
                    ax_list.append(ax)

                # Plot starts from 2nd column (1st column: time)
                ax_list[i].plot(tgrid, traj_data_history[epi_num, :, i + 1], colors[e])
                plt.xlabel('time', size=24)
                plt.xticks(fontsize=20)
                plt.ylabel(x_label[i], size=24)
                plt.yticks(fontsize=20)
                plt.grid()

            for i in range(self.a_dim):
                if e == 0:
                    ax = fig.add_subplot(2, 4, i + self.s_dim)
                    ax_list.append(ax)

                ax_list[i + self.s_dim - 1].plot(tgrid, traj_data_history[epi_num, :, i + self.s_dim + self.o_dim], colors[e])
                plt.xlabel('time', size=24)
                plt.xticks(fontsize=20)
                plt.ylabel(u_label[i], size=24)
                plt.yticks(fontsize=20)
                plt.grid()
        fig.tight_layout()
        plt.savefig(self.result_save_path + 'trajectory_plot.png')
        plt.show()

        fig = plt.figure(1, figsize=[20, 12])
        label = ['cost', 'loss']
        for i in range(2):
            ax = fig.add_subplot(1, 2, i + 1)
            ax.plot(stat_history[:, i])
            plt.xlabel('episode', size=24)
            plt.xticks(fontsize=20)
            plt.ylabel(label[i], size=24)
            plt.yticks(fontsize=20)
            plt.grid()

        fig.tight_layout()
        plt.savefig(self.result_save_path + 'stats_plot.png')
        plt.show()
