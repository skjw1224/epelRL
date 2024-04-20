import os
import numpy as np
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, config, env, agent):
        self.config = config
        self.agent = agent
        self.agent_name = config.algo
        self.env = env

        self.nT = self.env.nT
        self.max_episode = self.config.max_episode
        self.save_freq = self.config.save_freq
        self.plot_episode = [self.save_freq*(i+1)-1 for i in range(self.max_episode//self.save_freq)]

        self.save_path = self.config.save_path
        self.save_freq = self.config.save_freq

        self.learning_stat_lst = ['Cost'] + self.agent.loss_lst
        self.learning_stat_dim = len(self.learning_stat_lst)
        self.learning_stat_history = np.zeros((self.max_episode, self.learning_stat_dim))

        self.traj_dim = self.env.s_dim + self.env.a_dim + 1
        self.traj_data_history = np.zeros((self.max_episode, self.nT, self.traj_dim))

    def train(self):
        print('---------------------------------------')
        print(f'Environment: {self.config.env}, Algorithm: {self.agent_name}, Seed: {self.config.seed}, Device: {self.config.device}')
        print('---------------------------------------')

        if self.agent_name in ['DQN', 'QRDQN', 'DDPG', 'TD3', 'SAC', 'GDHP']:
            self._train_per_single_step()
        elif self.agent_name in ['A2C', 'TRPO', 'PPO', 'iLQR', 'SDDP', 'PoWER']:
            self._train_per_single_episode()
        elif self.agent_name in ['REPS', 'PI2']:
            self._train_per_multiple_episodes()

    def _train_per_single_step(self):
        for epi in range(self.max_episode):

            s, a = self.env.reset()
            for step in range(self.nT):
                a = self.agent.ctrl(s)

                if self.env.need_derivs:
                    s2, r, is_term, derivs = self.env.step(s, a)
                    self.agent.add_experience(s, a, r, s2, is_term, derivs)
                else:
                    s2, r, is_term = self.env.step(s, a)
                    self.agent.add_experience(s, a, r, s2, is_term)
                
                loss = self.agent.train()
                self.learning_stat_history[epi, :] += np.concatenate((r.reshape(1, ), loss))
                
                s_denorm = self.env.descale(s, self.env.xmin, self.env.xmax)
                a_denorm = self.env.descale(a, self.env.umin, self.env.umax)
                self.traj_data_history[epi, step, :] = np.concatenate((s_denorm, a_denorm, r), axis=0).squeeze()
                
                s = s2

            self.learning_stat_history[epi, :] /= self.nT
            self._print_stats(epi)

        self._save_history()

    def _train_per_single_episode(self):
        for epi in range(self.max_episode):
            
            epi_return = 0.
            s, a = self.env.reset()
            for step in range(self.nT):
                a = self.agent.ctrl(s)
                
                if self.env.need_derivs:
                    s2, r, is_term, derivs = self.env.step(s, a)
                    self.agent.add_experience(s, a, r, s2, is_term, derivs)
                else:
                    s2, r, is_term = self.env.step(s, a)
                    self.agent.add_experience(s, a, r, s2, is_term)

                epi_return += r.item()

                s_denorm = self.env.descale(s, self.env.xmin, self.env.xmax)
                a_denorm = self.env.descale(a, self.env.umin, self.env.umax)
                self.traj_data_history[epi, step, :] = np.concatenate((s_denorm, a_denorm, r), axis=0).squeeze()

                s = s2

            loss = self.agent.train()
            self.learning_stat_history[epi, :] += np.concatenate((np.array([epi_return/self.nT]), loss))

            self._print_stats(epi)

        self._save_history()

    def _train_per_multiple_episodes(self):
        for epi in range(self.max_episode):
            
            epi_return = 0.
            for rollout in range(self.config.num_rollout):
                s, a = self.env.reset()
                for step in range(self.nT):
                    a = self.agent.ctrl(s)
                    s2, r, is_term = self.env.step(s, a)
                    self.agent.add_experience(s, a, r, s2, is_term)

                    epi_return += r.item()

                    if rollout == 0:
                        s_denorm = self.env.descale(s, self.env.xmin, self.env.xmax)
                        a_denorm = self.env.descale(a, self.env.umin, self.env.umax)
                        self.traj_data_history[epi, step, :] = np.concatenate((s_denorm, a_denorm, r), axis=0).squeeze()

                    s = s2

            loss = self.agent.train()
            self.learning_stat_history[epi, :] += np.concatenate((np.array([epi_return/(self.nT*self.config.num_rollout)]), loss))

            self._print_stats(epi)
        
        self._save_history()

    def _print_stats(self, epi):
        print(f'Episode: {epi}')
        for i, stat_type in enumerate(self.learning_stat_lst):
            print(f'-- {stat_type}: {self.learning_stat_history[epi, i]:.8f}')
        print('---------------------------------------')

    def _save_history(self):
        np.save(os.path.join(self.save_path, 'learning_stat_history.npy'), self.learning_stat_history)
        np.save(os.path.join(self.save_path, 'traj_data_history.npy'), self.traj_data_history)

    def plot(self):
        self._plot_traj_data()
        self._plot_learning_stat()
    
    def _plot_traj_data(self):
        variable_tag_lst = self.env.plot_info['variable_tag_lst']
        ref_idx_lst = self.env.plot_info['ref_idx_lst']
        nrows_s, ncols_s = self.env.plot_info['state_plot_shape']
        nrows_a, ncols_a = self.env.plot_info['action_plot_shape']
        
        ref = self.env.ref_traj()
        x_axis = np.linspace(self.env.t0+self.env.dt, self.env.tT, num=self.env.nT)
        
        # State variables subplots
        fig1, ax1 = plt.subplots(nrows_s, ncols_s, figsize=(ncols_s*6, nrows_s*5))
        for i, idx in enumerate(ref_idx_lst):
            ax1.flat[idx-1].hlines(ref[i], self.env.t0, self.env.tT, color='r', linestyle='--', label='Set point')
        for i in range(self.env.s_dim - 1):
            ax1.flat[i].set_xlabel(variable_tag_lst[0])
            ax1.flat[i].set_ylabel(variable_tag_lst[i+1])
            for epi in self.plot_episode:
                ax1.flat[i].plot(x_axis, self.traj_data_history[epi, :, i+1], label=f'Episode {epi+1}')
            ax1.flat[i].legend()
            ax1.flat[i].grid()
        fig1.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'{self.env.env_name}_{self.agent_name}_state_traj.png'))
        plt.show()

        # Control variables subplots
        fig2, ax2 = plt.subplots(nrows=1, ncols=len(ref_idx_lst), figsize=(10, 6), squeeze=False)
        for i, idx in enumerate(ref_idx_lst):
            ax2[0, i].set_xlabel(variable_tag_lst[0])
            ax2[0, i].set_ylabel(variable_tag_lst[idx])
            for epi in self.plot_episode:
                ax2[0, i].plot(x_axis, self.traj_data_history[epi, :, idx], label=f'Episode {epi+1}')
            ax2[0, i].hlines(ref[i], self.env.t0, self.env.tT, color='r', linestyle='--', label='Set point')
            ax2[0, i].legend()
            ax2[0, i].grid()
        fig2.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'{self.env.env_name}_{self.agent_name}_CV_traj.png'))
        plt.show()

        # Action variables subplots
        x_axis = np.linspace(self.env.t0, self.env.tT, num=self.env.nT+1)
        fig3, ax3 = plt.subplots(nrows_a, ncols_a, figsize=(ncols_a*6, nrows_a*5))
        for i in range(self.env.a_dim):
            ax3.flat[i].set_xlabel(variable_tag_lst[0])
            ax3.flat[i].set_ylabel(variable_tag_lst[self.env.s_dim + i])
            for epi in self.plot_episode:
                ax3.flat[i].stairs(self.traj_data_history[epi, :, self.env.s_dim + i], x_axis, label=f'Episode {epi+1}')
            ax3.flat[i].legend()
            ax3.flat[i].grid()
        fig3.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'{self.env.env_name}_{self.agent_name}_action_traj.png'))
        plt.show()

    def _plot_learning_stat(self):
        if self.learning_stat_dim == 2:
            nrows, ncols, figsize = 1, 2, (10, 6)
        elif self.learning_stat_dim == 3:
            nrows, ncols, figsize = 1, 3, (13, 6)
        elif self.learning_stat_dim == 4:
            nrows, ncols, figsize = 2, 2, (13, 13)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i in range(self.learning_stat_dim):
            ax.flat[i].plot(self.learning_stat_history[:, i])
            ax.flat[i].set_xlabel('Episode', size=20)
            ax.flat[i].set_ylabel(self.learning_stat_lst[i], size=20)
            ax.flat[i].grid()
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'{self.env.env_name}_{self.agent_name}_stats_plot.png'))
        plt.show()

