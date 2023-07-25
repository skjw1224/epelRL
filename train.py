import numpy as np
import matplotlib.pyplot as plt
import pickle


class Train(object):
    def __init__(self, config):
        self.config = config
        self.controller = self.config.algorithm['controller']['function'](config)
        self.type = self.config.algorithm['controller']['type']
        self.env = self.config.environment

        self.s_dim = self.env.s_dim
        self.a_dim = self.env.a_dim
        self.o_dim = self.env.o_dim

        self.t0 = self.env.t0  # ex) 0
        self.tT = self.env.tT  # ex) 2
        self.nT = self.env.nT  # ex) dt:0.005 nT = 401

        # Hyperparameters
        self.algorithm_type = self.config
        self.max_episode = self.config.hyperparameters['max_episode']
        self.save_period = self.config.hyperparameters['save_period']
        self.plot_snapshot = self.config.hyperparameters['plot_snapshot']
        self.result_save_path = self.config.result_save_path
        self.save_model = self.config.save_model

        self.traj_data_history = []
        self.conv_stat_history = []

    def env_rollout(self):
        print('---------------------------------------')
        print(f'Environment: {self.env.env_name}, Algorithm: {self.config.algorithm["controller"]["name"]}')
        print('---------------------------------------')
        if self.type == 'single_train_per_single_step':
            self._train_per_single_step()
        elif self.type == 'single_train_per_single_episode':
            self._train_per_single_episode()
        elif self.type == 'single_train_per_multiple_episodes':
            self._train_per_multiple_episodes()

    def _train_per_single_step(self):
        for epi in range(self.max_episode):
            epi_reward = 0.
            epi_conv_stat = np.zeros(len(self.controller.loss_lst))
            epi_traj_data = []

            t, s, o, a = self.env.reset()
            for step in range(self.nT):
                a = self.controller.ctrl(epi, step, s, a)

                if self.config.algorithm['controller']['action_type'] == 'discrete':
                    a_val = self.controller.action_idx2mesh(a)
                else:
                    a_val = a

                t2, s2, o2, r, is_term, derivs = self.env.step(t, s, a_val)
                ref = self.env.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax).reshape([1, -1])

                if self.config.algorithm['controller']['model_requirement'] == 'model_based':
                    self.controller.add_experience(s, a, r, s2, is_term, derivs)
                else:
                    self.controller.add_experience(s, a, r, s2, is_term)

                loss = self.controller.train()
                t, s, o = t2, s2, o2

                epi_reward += r.item()
                epi_conv_stat += loss
                if epi % self.save_period == 0:
                    epi_traj_data.append([s, a_val, r, o, ref])

            self._append_stats(epi_reward, epi_conv_stat)
            self._print_stats(epi, epi_reward, epi_conv_stat)
            self._append_traj_data(epi_traj_data)

        self._save_history()

    def _train_per_single_episode(self):
        for epi in range(self.max_episode):
            epi_reward = 0.
            epi_conv_stat = np.zeros(len(self.controller.loss_lst))

            t, s, o, a = self.env.reset()
            for step in range(self.nT):
                a = self.controller.ctrl(epi, step, s, a)
                t2, s2, o2, r, is_term, derivs = self.env.step(t, s, a)

                if self.config.algorithm['controller']['model_requirement'] == 'model_based':
                    self.controller.add_experience(s, a, r, s2, is_term, derivs)
                else:
                    self.controller.add_experience(s, a, r, s2, is_term)

                t, s = t2, s2
                epi_reward += r.item()

            loss = self.controller.train()
            epi_conv_stat += loss

            self._print_stats(epi, epi_reward, epi_conv_stat)

    def _train_per_multiple_episodes(self):
        for epi in range(self.max_episode):
            print(f'Episode {epi}')
            epi_reward = 0.
            epi_conv_stat = np.zeros(len(self.controller.loss_lst))

            self.controller.sampling(epi)
            loss = self.controller.train(epi)

            epi_conv_stat += loss
            print(epi_conv_stat)

    def _append_stats(self, epi_reward, epi_conv_stat):
        epi_stats = [epi_reward]
        for loss in epi_conv_stat:
            epi_stats.append(loss)
        self.conv_stat_history.append(np.array(epi_stats))

    def _print_stats(self, epi_num, epi_reward, epi_conv_stat):
        print(f'Episode: {epi_num}')
        print(f'- Rewards: {epi_reward:.4f}')
        for i, loss_type in enumerate(self.controller.loss_lst):
            print(f'- {loss_type}: {epi_conv_stat[i]:.4f}')
        print('---------------------------------------')

    def _append_traj_data(self, epi_traj_data):
        if epi_traj_data:
            temp_lst = []
            for traj_data in epi_traj_data:
                s, a, r, o, ref = traj_data
                s = self.env.descale(s, self.env.xmin, self.env.xmax).reshape([1, -1])
                o = self.env.descale(o, self.env.ymin, self.env.ymax).reshape([1, -1])
                a = self.env.descale(a, self.env.umin, self.env.umax).reshape([1, -1])
                r = r.reshape([1, -1])
                ref = self.env.descale(ref, self.env.ymin, self.env.ymax).reshape([1, -1])
                temp_data = np.concatenate([s, a, r, o, ref], axis=1).reshape([1, -1])
                temp_lst.append(temp_data)
            self.traj_data_history.append(np.array(temp_lst).squeeze())
        else:
            pass

    def _save_history(self):
        conv_stat_history = np.array(self.conv_stat_history)
        traj_data_history = np.array(self.traj_data_history)

        with open(self.result_save_path + 'final_solution.pkl', 'wb') as fw:
            solution = [conv_stat_history, traj_data_history]
            pickle.dump(solution, fw)

    def plot(self):
        with open(self.result_save_path + 'final_solution.pkl', 'rb') as fr:
            solution = pickle.load(fr)
        conv_stat_history, traj_data_history = solution

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
            ax2[i].plot(conv_stat_history[:, i])
            ax2[i].set_xlabel('episode', size=20)
            ax2[i].set_ylabel(stat_label[i], size=20)
            ax2[i].grid()
        fig2.tight_layout()
        plt.savefig(self.result_save_path + 'stats_plot.png')
        plt.show()
