import os
import numpy as np
import matplotlib.pyplot as plt


class Tester(object):
    def __init__(self, config, env, agent):
        self.config = config
        self.save_path = self.config['save_path']

        self.agent = agent
        self.agent_name = config['algo']
        self.agent.load(self.save_path, 'converged_model')

        self.env = env

        self.nT = self.env.nT
        self.num_evaluate = self.config['num_test_evaluate']

        self.traj_dim = self.env.s_dim + self.env.a_dim + 1
        self.traj_data_history = np.zeros((self.num_evaluate, self.nT, self.traj_dim))
        self.episodic_avg_cost_history = np.zeros((self.num_evaluate, 1))

    def test(self):
        for eval_iter in range(self.num_evaluate):
            s, a = self.env.reset()
            episodic_avg_cost = 0.
            for step in range(self.nT):
                a = self.agent.ctrl(s)
                s2, r, _, _ = self.env.step(s, a)
                episodic_avg_cost += r

                s_denorm = self.env.descale(s, self.env.xmin, self.env.xmax)
                a_denorm = self.env.descale(a, self.env.umin, self.env.umax)
                self.traj_data_history[eval_iter, step, :] = np.concatenate((s_denorm, a_denorm, r),
                                                                                 axis=0).squeeze()

                s = s2

            episodic_avg_cost /= self.nT
            self.episodic_avg_cost_history[eval_iter,0] = episodic_avg_cost

        avg_cost, std_cost = np.mean(self.episodic_avg_cost_history), np.std(self.episodic_avg_cost_history)
        print(f'Test in {self.config["env"]} by {self.agent_name}')
        print(f'-- Average cost: {avg_cost:.8f}')
        print(f'-- Standard deviation of cost: {std_cost:.8f}')
        print('---------------------------------------')
        self._save_test_history()
        return avg_cost, std_cost

    def plot(self):
        self._plot_traj_data()

    def _save_test_history(self):
        np.save(os.path.join(self.save_path, 'test_traj_data_history.npy'), self.traj_data_history)
        np.save(os.path.join(self.save_path, 'test_cost_history.npy'), self.episodic_avg_cost_history)

    def _plot_traj_data(self):
        variable_tag_lst = self.env.plot_info['variable_tag_lst']
        state_plot_idx_lst = self.env.plot_info[
            'state_plot_idx_lst'] if 'state_plot_idx_lst' in self.env.plot_info else range(1, self.env.s_dim)
        ref_idx_lst = self.env.plot_info['ref_idx_lst']
        nrows_s, ncols_s = self.env.plot_info['state_plot_shape']
        nrows_a, ncols_a = self.env.plot_info['action_plot_shape']

        ref = self.env.ref_traj()
        x_axis = np.linspace(self.env.t0 + self.env.dt, self.env.tT, num=self.env.nT)

        traj_mean = self.traj_data_history.mean(axis=0)
        traj_std = self.traj_data_history.std(axis=0)

        # State variables subplots
        fig1, ax1 = plt.subplots(nrows_s, ncols_s, figsize=(ncols_s * 6, nrows_s * 5))
        for i, fig_idx in enumerate(ref_idx_lst):
            ax1.flat[fig_idx - 1].hlines(ref[i], self.env.t0, self.env.tT, color='r', linestyle='--', label='Set point')

        for fig_idx, i in enumerate(state_plot_idx_lst):
            ax1.flat[fig_idx].set_xlabel(variable_tag_lst[0])
            ax1.flat[fig_idx].set_ylabel(variable_tag_lst[fig_idx + 1])
            ax1.flat[fig_idx].plot(x_axis, traj_mean[:, i], label=f'{self.agent_name}')
            ax1.flat[fig_idx].fill_between(x_axis, traj_mean[:, i] + traj_std[:, i],
                                           traj_mean[:, i] - traj_std[:, i], alpha=0.5)
            ax1.flat[fig_idx].legend()
            ax1.flat[fig_idx].grid()
        fig1.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'test_{self.env.env_name}_{self.agent_name}_state_traj.png'))
        plt.show()
        plt.close()

        # # Controlled variables subplots
        # if len(ref_idx_lst) > 0:
        #     fig2, ax2 = plt.subplots(nrows=1, ncols=len(ref_idx_lst), figsize=(10, 6), squeeze=False)
        #     for i, idx in enumerate(ref_idx_lst):
        #         ax2[0, i].set_xlabel(variable_tag_lst[0])
        #         ax2[0, i].set_ylabel(variable_tag_lst[idx])
        #         for epi in self.plot_episode:
        #             ax2[0, i].plot(x_axis, traj_mean[epi, :, idx], label=f'Episode {epi + 1}')
        #             ax2[0, i].fill_between(x_axis, traj_mean[epi, :, idx]+traj_std[epi, :, idx], traj_mean[epi, :, idx]-traj_std[epi, :, idx], alpha=0.5)
        #         ax2[0, i].hlines(ref[i], self.env.t0, self.env.tT, color='r', linestyle='--', label='Set point')
        #         ax2[0, i].legend()
        #         ax2[0, i].grid()
        #     fig2.tight_layout()
        #     plt.savefig(os.path.join(self.save_path, f'{self.env.env_name}_{self.agent_name}_CV_traj.png'))
        #     plt.show()

        # Action variables subplots
        x_axis = np.linspace(self.env.t0, self.env.tT, num=self.env.nT)
        fig3, ax3 = plt.subplots(nrows_a, ncols_a, figsize=(ncols_a * 6, nrows_a * 5))
        for i in range(self.env.a_dim):
            axis = ax3.flat[i] if self.env.a_dim > 1 else ax3
            axis.set_xlabel(variable_tag_lst[0])
            axis.set_ylabel(variable_tag_lst[len(state_plot_idx_lst) + 1])
            axis.plot(x_axis, traj_mean[:, self.env.s_dim + i], label=f'{self.agent_name}')
            axis.fill_between(x_axis, traj_mean[:, self.env.s_dim + i] + traj_std[:, self.env.s_dim + i],
                              traj_mean[:, self.env.s_dim + i] - traj_std[:, self.env.s_dim + i],
                              alpha=0.5)
            axis.legend()
            axis.grid()
        fig3.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'test_{self.env.env_name}_{self.agent_name}_action_traj.png'))
        plt.show()
        plt.close()
