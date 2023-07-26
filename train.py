import os
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
            epi_traj_data = []

            t, s, o, a = self.env.reset()
            for step in range(self.nT):
                a = self.controller.ctrl(epi, step, s, a)
                t2, s2, o2, r, is_term, derivs = self.env.step(t, s, a)
                ref = self.env.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax).reshape([1, -1])

                if self.config.algorithm['controller']['model_requirement'] == 'model_based':
                    self.controller.add_experience(s, a, r, s2, is_term, derivs)
                else:
                    self.controller.add_experience(s, a, r, s2, is_term)

                t, s, o = t2, s2, o2
                epi_reward += r.item()
                if epi % self.save_period == 0:
                    epi_traj_data.append([s, a, r, o, ref])

            loss = self.controller.train()
            epi_conv_stat += loss

            self._append_stats(epi_reward, epi_conv_stat)
            self._print_stats(epi, epi_reward, epi_conv_stat)
            self._append_traj_data(epi_traj_data)

        self._save_history()

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
        print(f'- Cost: {epi_reward:.4f}')
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

        # State and action subplots
        self.env.plot_trajectory(traj_data_history, self.result_save_path)

        # Cost and loss subplots
        self._plot_conv_stat(conv_stat_history, self.result_save_path)

    def _plot_conv_stat(self, conv_stat_history, save_path):
        variable_tag = ['Cost']
        for loss in self.controller.loss_lst:
            variable_tag.append(loss)

        num_loss = len(variable_tag)
        if num_loss == 2:
            nrows, ncols, figsize = 1, 2, (10, 6)
        elif num_loss == 3:
            nrows, ncols, figsize = 1, 3, (13, 6)
        elif num_loss == 4:
            nrows, ncols, figsize = 2, 2, (13, 13)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i in range(num_loss):
            ax.flat[i].plot(conv_stat_history[:, i])
            ax.flat[i].set_xlabel('Episode', size=20)
            ax.flat[i].set_ylabel(variable_tag[i], size=20)
            ax.flat[i].grid()
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, 'stats_plot.png'))
        plt.show()

