import os
import numpy as np
import matplotlib.pyplot as plt
import time

from utility.pid import PID
from utility.custom_init_ctrl import InitCtrl
from config import plot_traj_data


class Trainer(object):
    def __init__(self, config, env, agent):
        self.config = config
        self.agent = agent
        self.agent_name = config['algo']
        self.env = env
        self.disp_opt = config['disp_opt']

        self.nT = self.env.nT
        self.max_episode = self.config['max_episode']
        self.save_freq = self.config['save_freq']
        self.plot_episode = [0] + [self.save_freq*(i+1)-1 for i in range(self.max_episode//self.save_freq)]
        self.warm_up_episode = self.config['warm_up_episode']
        self.num_evaluate = self.config['num_evaluate']

        self.save_path = self.config['save_path']
        self.save_freq = self.config['save_freq']
        self.show_plot = self.config['show_plot']

        self.learning_stat_lst = ['Cost', 'Convergence criteria'] + self.agent.loss_lst
        self.learning_stat_dim = len(self.learning_stat_lst)
        self.learning_stat_history = np.zeros((self.max_episode, self.learning_stat_dim))
        self.convg_bound = config['convg_bound']
        self.convg_scaling = None

        self.traj_dim = self.env.s_dim + self.env.a_dim + 1
        self.traj_data_history = np.zeros((self.num_evaluate, self.max_episode, self.nT, self.traj_dim))

    def train(self, start_time):
        # print('---------------------------------------')
        # print(f'Environment: {self.config.env}, Algorithm: {self.agent_name}, Seed: {self.config.seed}, Device: {self.config.device}')
        # print('---------------------------------------')

        self._warm_up_data()

        if self.agent_name in ['DQN', 'QRDQN', 'DDPG', 'TD3', 'SAC', 'GDHP']:
            self._train_per_single_step()
        elif self.agent_name in ['A2C', 'TRPO', 'PPO', 'iLQR', 'SDDP', 'PoWER']:
            self._train_per_single_episode()
        elif self.agent_name in ['REPS', 'PI2']:
            self._train_per_multiple_episodes()

        episodic_cpu_time = (time.time() - start_time) / np.size(self.traj_data_history, 1)
        np.save(os.path.join(self.save_path, 'computation_time.npy'), episodic_cpu_time)
    
    def _warm_up_data(self):
        if hasattr(self.env, 'pid_gain'):
            init_controller = PID()
            init_controller.set_info = {'o_dim': self.env.o_dim, 'a_dim': self.env.a_dim, 'dt': self.env.dt}
            init_controller.set_gain = self.env.pid_gain()
            init_controller.set_reference = self.env.scale(self.env.ref_traj(), self.env.ymin, self.env.ymax)
        elif hasattr(self.env, 'init_controller'):
            init_controller = InitCtrl()
            init_controller.set_controller = self.env.init_controller

        for epi in range(self.warm_up_episode):
            init_controller.reset()
            s, a = self.env.reset()

            for step in range(self.nT):
                o = self.env.get_observ(s, a)
                a = init_controller.ctrl(o)

                a = self.env.scale(a, self.env.umin, self.env.umax)

                s2, r, is_term, derivs = self.env.step(s, a)
                self.agent.add_experience((s, a, r, s2, is_term, derivs))

                s = s2

    def _train_per_single_step(self):
        for epi in range(self.max_episode):

            s, a = self.env.reset()
            for step in range(self.nT):
                a = self.agent.ctrl(s)

                s2, r, is_term, derivs = self.env.step(s, a)
                self.agent.add_experience((s, a, r, s2, is_term, derivs))

                self._single_train_and_stat_update(epi)

                s = s2

            self._update_convg_criteria(epi)
            self._evaluate(epi)
            self._print_stats(epi)
            if self._check_if_converged(epi):
                break

        self._save_history()

    def _train_per_single_episode(self):
        for epi in range(self.max_episode):
            s, a = self.env.reset()
            for step in range(self.nT):
                a = self.agent.ctrl(s)
                
                s2, r, is_term, derivs = self.env.step(s, a)
                self.agent.add_experience((s, a, r, s2, is_term, derivs))

                s = s2

            self._single_train_and_stat_update(epi)

            self._update_convg_criteria(epi)
            self._evaluate(epi)
            self._print_stats(epi)
            if self._check_if_converged(epi):
                break

        self._save_history()

    def _train_per_multiple_episodes(self):
        for epi in range(self.max_episode):
            for rollout in range(self.config['num_rollout']):
                s, a = self.env.reset()
                for step in range(self.nT):
                    a = self.agent.ctrl(s)
                    s2, r, is_term, derivs = self.env.step(s, a)
                    self.agent.add_experience((s, a, r, s2, is_term, derivs))

                    s = s2

            self._single_train_and_stat_update(epi)

            self._update_convg_criteria(epi)
            self._evaluate(epi)
            self._print_stats(epi)
            if self._check_if_converged(epi):
                break
        
        self._save_history()

    def _evaluate(self, epi):
        avg_cost = 0.
        for eval_iter in range(self.num_evaluate):
            s, a = self.env.reset()
            for step in range(self.nT):
                a = self.agent.ctrl(s)
                s2, r, _, _ = self.env.step(s, a)
                avg_cost += r

                s_denorm = self.env.descale(s, self.env.xmin, self.env.xmax)
                a_denorm = self.env.descale(a, self.env.umin, self.env.umax)
                self.traj_data_history[eval_iter, epi, step, :] = np.concatenate((s_denorm, a_denorm, r), axis=0).squeeze()

                s = s2

        avg_cost /= (self.nT * self.num_evaluate)
        self.learning_stat_history[epi, 0] = avg_cost

    def _print_stats(self, epi):
        if self.disp_opt:
            print(f'Episode: {epi} - by {self.agent_name} in {self.config["env"]}')
            for i, stat_type in enumerate(self.learning_stat_lst):
                print(f'-- {stat_type}: {self.learning_stat_history[epi, i]:.8f}')
            print('---------------------------------------')

    def _save_history(self):
        np.save(os.path.join(self.save_path, 'learning_stat_history.npy'), self.learning_stat_history)
        np.save(os.path.join(self.save_path, 'traj_data_history.npy'), self.traj_data_history)

    def plot(self):
        case_name = [f'Episode {epi + 1}' for epi in range(self.max_episode)]
        save_name = os.path.join(self.save_path, f'{self.env.env_name}_{self.agent_name}')
        plot_traj_data(self.env, self.traj_data_history, self.plot_episode, case_name, save_name, self.show_plot)
        self._plot_learning_stat()

    def _plot_learning_stat(self):
        if self.learning_stat_dim <= 2:
            nrows, ncols, figsize = 1, 2, (10, 6)
        elif self.learning_stat_dim == 3:
            nrows, ncols, figsize = 1, 3, (13, 6)
        elif self.learning_stat_dim == 4:
            nrows, ncols, figsize = 2, 2, (13, 13)
        elif self.learning_stat_dim <= 6:
            nrows, ncols, figsize = 2, 3, (18, 13)
        elif self.learning_stat_dim <= 8:
            nrows, ncols, figsize = 2, 4, (23, 13)
        elif self.learning_stat_dim <= 9:
            nrows, ncols, figsize = 3, 3, (18, 18)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for i in range(self.learning_stat_dim):
            ax.flat[i].plot(self.learning_stat_history[:, i])
            ax.flat[i].set_xlabel('Episode', size=20)
            ax.flat[i].set_ylabel(self.learning_stat_lst[i], size=20)
            ax.flat[i].grid()
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'{self.env.env_name}_{self.agent_name}_stats_plot.png'))
        if self.show_plot:
            plt.show()
        plt.close()

    def _update_convg_criteria(self, epi):
        if epi < 1:
            self.learning_stat_history[epi, 1] = np.NAN
        elif epi < 2:
            self.convg_scaling = np.std(self.learning_stat_history[max(0, epi+1 - 50):epi+1, 2])
            self.learning_stat_history[epi, 1] = 1.
        else:
            self.learning_stat_history[epi, 1] = \
                np.std(self.learning_stat_history[max(0, epi+1 - 50):epi+1, 2]) / self.convg_scaling

    def _trim_histories(self, epi):
        self.learning_stat_history = np.delete(self.learning_stat_history, slice(epi + 1, self.max_episode), 0)
        self.traj_data_history = np.delete(self.traj_data_history, slice(epi + 1, self.max_episode), 1)
        self.plot_episode = [i for i in self.plot_episode if i < epi+1]

    def _single_train_and_stat_update(self, epi):
        loss = self.agent.train()
        self.learning_stat_history[epi, 2:] += loss

    def _check_if_converged(self, epi):
        if_convg = (self.learning_stat_history[epi,1] < self.convg_bound)
        if if_convg:
            print(f"Converged at epi {epi} - {self.agent_name}")
            self._trim_histories(epi)
        return if_convg

    def get_train_results(self):
        costs = self.learning_stat_history[:, 0]
        minimum_cost = np.min(costs)

        return minimum_cost

    def save_model(self):
        self.agent.save(self.save_path, 'converged_model')
