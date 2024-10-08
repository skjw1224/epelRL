import os
import numpy as np
import matplotlib.pyplot as plt
from config import plot_traj_data


class Tester(object):
    def __init__(self, config, env, agent):
        self.config = config
        self.load_path = self.config['load_path']
        self.save_path = self.config['save_path']
        self.show_plot = self.config['show_plot']

        self.agent = agent
        self.agent_name = config['algo']
        self.agent.load(self.load_path, 'converged_model')

        self.env = env

        self.nT = self.env.nT
        self.num_evaluate = self.config['num_test_evaluate']

        self.traj_dim = self.env.s_dim + self.env.a_dim + 1
        self.traj_data_history = np.zeros((self.num_evaluate, 1, self.nT, self.traj_dim))
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
                self.traj_data_history[eval_iter, 0, step, :] = np.concatenate((s_denorm, a_denorm, r),
                                                                                 axis=0).squeeze()

                s = s2

            episodic_avg_cost /= self.nT
            self.episodic_avg_cost_history[eval_iter,0] = episodic_avg_cost

        avg_cost, std_cost = np.mean(self.episodic_avg_cost_history), np.std(self.episodic_avg_cost_history)
        print(f'-- Average cost: {avg_cost:.8f}')
        print(f'-- Standard deviation of cost: {std_cost:.8f}')
        print('---------------------------------------')
        self._save_test_history()
        return self.traj_data_history

    def plot(self):
        case_name = ['Test']
        save_name = os.path.join(self.save_path, f'{self.env.env_name}_{self.agent_name}')
        plot_traj_data(self.env, self.traj_data_history, [0], case_name, save_name, self.show_plot)

    def _save_test_history(self):
        np.save(os.path.join(self.save_path, 'test_traj_data_history.npy'), self.traj_data_history)
        np.save(os.path.join(self.save_path, 'test_cost_history.npy'), self.episodic_avg_cost_history)
