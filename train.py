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
        self.rollout_iter = self.config.hyperparameters['rollout_iter']

        self.traj_data_history = []
        self.stat_history = []

    def env_rollout2(self):
        for epi in range(self.max_episode + 1):
            epi_reward = 0.
            for _ in range(self.rollout_iter):
                # Initialize
                t, x, y, u = self.env.reset()
                for i in range(self.nT):
                    u = self.controller.ctrl(epi, i, x, u)
                    t2, x2, y2, r, is_term, _ = self.env.step(t, x, u)
                    ref = self.env.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax).reshape([1, -1])
                    self.controller.add_experience(x, u, r, x2, is_term)
                    epi_reward += r.item()
                    t, x = t2, x2
            loss = self.controller.train()

            print(f'Episode {epi+1}')
            print(f'Loss: {loss}')
            print(f'Reward: {epi_reward / (self.nT*self.rollout_iter)}')

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

        # state and action subplots
        fig1, ax1 = plt.subplots(nrows=2, ncols=4, figsize=(20, 12))
        fig1.subplots_adjust(hspace=.4, wspace=.5)
        traj_label = [r'$C_{A}[mol/L]$', r'$C_{B}[mol/L]$', r'$T_{R}[C]$', r'$T_{C}[C]$',
                      r'$\frac{\dot{V}}{V_{R}}[h^{-1}]$', r'$\dot{Q}[kJ/h]$',
                      r'$\frac{\Delta\dot{V}}{V_{R}}[h^{-1}]$', r'$\Delta\dot{Q}[kJ/h]$']
        colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
        ref = self.env.ref_traj()
        for i in range(self.s_dim + self.a_dim - 1):
            for epi_num in self.plot_snapshot:
                epi_num = int(epi_num / self.save_period)
                time_grid = traj_data_history[epi_num, :, 0]
                ax1.flat[i].plot(time_grid, traj_data_history[epi_num, :, i + 1], colors[epi_num], label=epi_num)
                ax1.flat[i].set_xlabel('time[h]', fontsize=15)
                ax1.flat[i].set_ylabel(traj_label[i], fontsize=15)
                ax1.flat[i].legend()
                ax1.flat[i].grid()
        ax1.flat[1].plot(time_grid, ref[0]*np.ones((self.env.nT, 1)), 'r--', label='set point')
        ax1.flat[1].legend()
        fig1.tight_layout()
        plt.savefig(self.result_save_path + 'trajectory_plot.png')
        plt.show()

        # cost and loss subplots
        fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(20, 12))
        stat_label = ['Cost', 'Loss']
        for i in range(2):
            ax2[i].plot(stat_history[:, i])
            ax2[i].set_xlabel('episode', size=20)
            ax2[i].set_ylabel(stat_label[i], size=20)
            ax2[i].grid()
        fig2.tight_layout()
        plt.savefig(self.result_save_path + 'stats_plot.png')
        plt.show()
