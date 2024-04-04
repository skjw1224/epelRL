import os
import numpy as np
import matplotlib.pyplot as plt
import pickle


class Trainer(object):
    def __init__(self, config, env, agent):
        self.config = config
        self.agent = agent
        self.agent_name = config.algo
        self.env = env

        self.nT = self.env.nT
        self.max_episode = self.config.max_episode
        self.plot_episode = [10*(i+1)-1 for i in range(self.max_episode//10)]

        self.result_save_path = self.config.result_save_path
        self.model_save_path = self.config.model_save_path
        self.save_freq = self.config.save_freq

        self.learning_stat_lst = ['Cost'] + self.agent.loss_lst
        self.learning_stat_dim = len(self.learning_stat_lst)
        self.learning_stat_history = np.zeros((self.max_episode, self.learning_stat_dim))

        self.traj_dim = self.env.s_dim + self.env.a_dim + 1 + self.env.o_dim
        self.traj_data_history = np.zeros((self.max_episode, self.nT, self.traj_dim))

    def train(self):
        print('---------------------------------------')
        print(f'Environment: {self.config.env}, Algorithm: {self.agent_name}, Seed: {self.config.seed}, Device: {self.config.device}')
        print('---------------------------------------')

        if self.agent_name in ['DQN', 'QRDQN', 'DDPG', 'SAC', 'GDHP']:
            self._train_per_single_step()
        elif self.agent_name in ['A2C', 'TRPO', 'PPO', 'iLQR', 'SDDP']:
            self._train_per_single_episode()
        elif self.agent_name in ['REPS', 'PoWER', 'PI2']:
            self._train_per_multiple_episodes()

    def _train_per_single_step(self):
        for epi in range(self.max_episode):

            t, s, o, a = self.env.reset()
            for step in range(self.nT):
                a = self.agent.ctrl(s)
                t2, s2, o2, r, is_term, derivs = self.env.step(t, s, a)
                self.agent.add_experience(s, a, r, s2, is_term)
                
                loss = self.agent.train()
                self.learning_stat_history[epi, :] += np.concatenate((r.reshape(1, ), loss))
                
                s_denorm = self.env.descale(s, self.env.xmin, self.env.xmax)
                o_denorm = self.env.descale(o, self.env.ymin, self.env.ymax)
                a_denorm = self.env.descale(a, self.env.umin, self.env.umax)
                self.traj_data_history[epi, step, :] = np.concatenate((s_denorm, a_denorm, r, o_denorm), axis=0).squeeze()
                
                t, s, o = t2, s2, o2

            self.learning_stat_history[epi, :] /= self.nT
            self._print_stats(epi)

        self._save_history()

    def _train_per_single_episode(self):
        for epi in range(self.max_episode):

            t, s, o, a = self.env.reset()
            for step in range(self.nT):
                a = self.agent.ctrl(s)
                t2, s2, o2, r, is_term, derivs = self.env.step(t, s, a)
                self.agent.add_experience(s, a, r, s2, is_term)

                s_denorm = self.env.descale(s, self.env.xmin, self.env.xmax)
                o_denorm = self.env.descale(o, self.env.ymin, self.env.ymax)
                a_denorm = self.env.descale(a, self.env.umin, self.env.umax)
                self.traj_data_history[epi, step, :] = np.concatenate((s_denorm, a_denorm, r, o_denorm), axis=0).squeeze()

                t, s, o = t2, s2, o2

            loss = self.agent.train()
            self.learning_stat_history[epi, :] += np.concatenate((r.reshape(1, ), loss))

            self._print_stats(epi)

        self._save_history()

    def _train_per_multiple_episodes(self):
        for epi in range(self.max_episode):
            print(f'Episode {epi}')
            epi_return = 0.
            epi_conv_stat = np.zeros(len(self.agent.loss_lst))

            self.agent.sampling(epi)
            loss = self.agent.train(epi)

            epi_conv_stat += loss
            print(epi_conv_stat)

    def _print_stats(self, epi):
        print(f'Episode: {epi}')
        for i, stat_type in enumerate(self.learning_stat_lst):
            print(f'-- {stat_type}: {self.learning_stat_history[epi, i]:.8f}')
        print('---------------------------------------')

    def _save_history(self):
        np.save(os.path.join(self.result_save_path, 'learning_stat_history.npy'), self.learning_stat_history)
        np.save(os.path.join(self.result_save_path, 'traj_data_history.npy'), self.traj_data_history)

    def plot(self):
        with open(self.result_save_path + 'final_solution.pkl', 'rb') as fr:
            solution = pickle.load(fr)
        learning_stat_history, traj_data_history = solution

        # State and action subplots
        self.env.plot_trajectory(traj_data_history, self.plot_episode, self.agent_name, self.result_save_path)

        # Cost and loss subplots
        self._plot_conv_stat(learning_stat_history, self.result_save_path)

    def _plot_conv_stat(self, learning_stat_history, save_path):
        variable_tag = ['Cost']
        for loss in self.agent.loss_lst:
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
            ax.flat[i].plot(learning_stat_history[:, i])
            ax.flat[i].set_xlabel('Episode', size=20)
            ax.flat[i].set_ylabel(variable_tag[i], size=20)
            ax.flat[i].grid()
        fig.tight_layout()
        plt.savefig(os.path.join(save_path, f'{self.env.env_name}_{self.agent_name}_stats_plot.png'))
        plt.show()

