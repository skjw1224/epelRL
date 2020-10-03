import numpy as np
import matplotlib.pyplot as plt

class DataPostProcessing(object):
    def __init__(self, config):
        self.config = config
        self.env = config.environment
        self.save_period = config.hyperparameters['save_period']

        self.s_dim = self.env.s_dim
        self.o_dim = self.env.o_dim
        self.a_dim = self.env.a_dim

        self.nT = self.env.nT

        # Stat history: Mean Episode value of cost
        self.epi_stat_history = []
        # Path data history: x, y, u, r, ref
        self.epi_path_data_history = []
        # Term data history: xT, yT, rT
        self.epi_term_data_history = []

    def stats_record(self, epi_solutions, epi_data, epi_misc_data, epi_num):
        """Args: epi CDDP solutions: CDDP solutions --->  Vlist, gain_list, path (u, l), leg (m), term (mT), master (s, n), hyperparameters, convergence stat
                 epi data: Multiple shooting rollout data --->  x, x2, path (u, l), leg (m), term (mT), master (s, n), hyperparameters, systemparameters
                 epi misc data: Multiple shooting miscellaneous data --> r, y, constraints

           Output: stat history ---> Augmented cost (or Value functions)  &  Convergence Jac norm
                   data history ---> x, y, u, r, const, Lagr, opt par, V list"""

        epi_path_data, epi_term_data = epi_data
        epi_path_misc_data, epi_term_misc_data = epi_misc_data


        if epi_solutions is not None:
            epi_path_solution, epi_term_solution = epi_solutions

        else: # Initial iteration --> No CDDP solution --> Assign null vars to path/leg/term/master solutions
            V_zeros = [[np.zeros([1, self.V_dim])] for _ in range(self.nT - 1)]
            VT_zeros = [[np.zeros([1, self.VT_dim])] for _ in range(1)]

            epi_path_solution = [V_zeros, V_zeros, None, None, None, None, np.zeros([self.nT - 1, 1]), np.zeros([self.nT - 1, 1])]
            epi_term_solution = [VT_zeros, VT_zeros, None, None, None, np.zeros([1, 1])]


        "Stat history"
        # Path stat history
        Vn_list_epi, Vn_woa_list_epi, path_gain_list_epi, _, _, _, cost_aug_path_epi, conv_stat_path_epi = epi_path_solution
        epi_path_stat = np.array([[np.mean(cost_aug_path_epi), np.mean(conv_stat_path_epi)]])

        # terminal stat history
        VnT_list_epi, VnT_woa_list_epi, term_gain_list_epi, _, _, conv_stat_term_epi = epi_term_solution
        epi_term_stat = np.array([[np.mean([_[0] for _ in VnT_list_epi]), np.mean(conv_stat_term_epi)]])

        # Save episode convergence statistics
        epi_stat = np.concatenate([epi_path_stat, epi_term_stat], axis=1)
        self.epi_stat_history.append(epi_stat)


        if epi_num % self.save_period == 0:
            "Data history"
            # Path data history
            for path_data, path_misc_data in zip(epi_path_data, epi_path_misc_data):
                x, x2, u = path_data
                r, y2, ref = path_misc_data

                x_record = np.reshape(x, [1, self.s_dim])
                y_record = np.reshape(y2, [1, self.o_dim])
                u_record = np.reshape(u, [1, self.a_dim])
                r_record = np.reshape(r, [1, -1])
                ref_record = np.reshape(ref, [1, -1])

                temp_data_history = np.concatenate([x_record, y_record, u_record, r_record, ref_record], 1).reshape([1, -1])

                self.epi_path_data_history.append(temp_data_history)

           # terminal data history
            for term_data, term_misc_data in zip(epi_term_data, epi_term_misc_data):
                xT = term_data
                rT, yT = term_misc_data

                xT_record = np.reshape(xT, [1, self.s_dim])
                yT_record = np.reshape(yT, [1, self.o_dim])
                rT_record = np.reshape(rT, [1, 1])

                temp_data_history = np.concatenate([xT_record, yT_record, rT_record], 1).reshape([1, -1])

                self.epi_term_data_history.append(temp_data_history)


    def print_and_save_history(self, epi_num, prefix=None):
        if prefix is None:
            prefix = str('')

        np.set_printoptions(precision=4)
        print('| Episode ', '| Cost ', '| Conv ', '| Term cost ', '| Term conv ')
        print(epi_num,
              np.array2string(self.epi_stat_history[-1, :], formatter={'float_kind': lambda x: "    %.4f" % x}))

        np.savetxt('stat_history.txt', self.epi_stat_history, newline='\n')
        if epi_num % self.save_period == 0:
            np.savetxt('result/' + prefix + '_path_data_history.txt', self.epi_path_data_history, newline='\n')
            np.savetxt('result/' + prefix + '_term_data_history.txt', self.epi_term_data_history, newline='\n')


    def plot(self, epi_num):
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        fig = plt.figure(figsize=[20, 12])
        fig.subplots_adjust(hspace=.4, wspace=.5)
        label = [r'$C_{A}$', r'$C_{B}$', r'$T$', r'$T_{Q}$', r'$\frac{v}{V_{R}}$', r'$Q$',
                 r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$', r'$C_{B}$', r'$cost$']
        for j in range(len(label)):
            if label[j] in (r'$\frac{\Delta v}{V_{R}}$', r'$\Delta Q$'):
                ax = fig.add_subplot(2, 6, j + 5)
            else:
                ax = fig.add_subplot(2, 6, j + 1)
            ax.plot(trajectory[1:, j + 1])
            if j in (1, 8):
                ax.plot(trajectory[1:, -1], ':g')
            plt.ylabel(label[j], size=8)
        plt.savefig('result/episode' + str(epi_num) + '.png')
        plt.show()