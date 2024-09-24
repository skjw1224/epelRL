import os
import numpy as np
import casadi as ca
import scipy as sp
import matplotlib.pyplot as plt

from .base_environment import Environment

# Physio-chemical parameters for the polymerization reactor
R = 8.314           # gas constant [kJ/mol/K]
T_F = 25 + 273.15   # feed temperature [K]
E_a = 8500.0        # activation energy [kJ/mol]
delH_R = 950.0 * 1.00  # sp reaction enthalpy [kJ/kg]
A_tank = 65.0       # area heat exchanger surface jacket [m2]

k_0 = 7.0 * 1.00    # sp reaction rate
k_U2 = 32.0         # reaction parameter 1
k_U1 = 4.0          # reaction parameter 2
w_WF = .333         # mass fraction water in feed
w_AF = .667         # mass fraction of A in feed

m_M_KW = 5000.0     # mass of coolant in jacket [kg]
fm_M_KW = 3E5       # coolant flow in jacket [kg/h]
m_AWT_KW = 1000.0   # mass of coolant in EHE [kg]
fm_AWT_KW = 1E5     # coolant flow in EHE [kg/h]
m_AWT = 200.0       # mass of product in EHE [kg]
fm_AWT = 20000.0    # product flow in EHE [kg/h]
m_S = 39000.0       # mass of reactor steel [kg]

c_pW = 4.2          # sp heat cap coolant [kJ/kg/K]
c_pS = .47          # sp heat cap steel [kJ/kg/K]
c_pF = 3.0          # sp heat cap feed [kJ/kg/K]
c_pR = 5.0          # sp heat cap reactor contents [kJ/kg/K]

k_WS = 17280.0      # heat transfer coeff water-steel [W/m2/K]
k_AS = 3600.0       # heat transfer coeff monomer-steel [W/m2/K]
k_PS = 360.0        # heat transfer coeff product-steel [W/m2/K]

alfa = 5 * 20e4 * 3.6 # experimental coefficient [/s]

p_1 = 1.0

# Parameters with uncertainty
delH_R = 950.0      # specific reaction enthalpy [kJ/kg]
k_0 = 7.0           # specific reaction rate

class POLYMER(Environment):
    def __init__(self, config):
        self.env_name = 'POLYMER'
        self.config = config

        # Uncertain parameters: delH_R, k_0
        self.param_real = np.array([[delH_R, k_0]]).T
        self.param_range = np.array([[delH_R * 0.3, k_0 * 0.3]]).T
        self.p_mu = self.param_real
        self.p_sigma = np.zeros([np.shape(self.param_real)[0], 1])
        self.p_eps = np.zeros([np.shape(self.param_real)[0], 1])
        self.param_uncertainty = False
        self.param_extreme = False

        # Dimension
        self.s_dim = 14
        self.a_dim = 3
        self.o_dim = 2
        self.p_dim = np.shape(self.param_real)[0]

        self.t0 = 0.
        self.dt = 60 / 3600
        self.tT = 1.6

        m_W0 = 10000.0
        m_A0 = 853.0
        m_P0 = 26.5

        T_R0 = 90.0 + 273.15
        T_S0 = 90.0 + 273.15
        Tout_M0 = 90.0 + 273.15
        T_EK0 = 35.0 + 273.15
        Tout_AWT0 = 35.0 + 273.15
        accum_monom0 = 300.0
        T_adiab0 = m_A0 * delH_R / ((m_W0 + m_A0 + m_P0) * c_pR) + T_R0
        m_dot_f0 = 0.
        T_in_M0 = 60. + 273.15
        T_in_EK0 = 60. + 273.15

        # state: t, m_W, m_A, m_P, T_R, T_S, Tout_M, T_EK, Tout_AWT, accum_monom, T_adiab, m_dot_f, T_in_M, T_in_EK
        # action: dm_dot_f, dT_in_M, dT_in_EK
        # observation: t, m_P

        self.x0 = np.array([[self.t0, m_W0, m_A0, m_P0, T_R0, T_S0, Tout_M0, T_EK0, Tout_AWT0, accum_monom0, T_adiab0, m_dot_f0, T_in_M0, T_in_EK0]]).T
        self.u0 = np.array([[0., 0., 0.]]).T
        self.nT = int(self.tT / self.dt)  # episode length

        self.xmin = np.array([[self.t0, 0., 0., 0., 298, 298, 298, 288, 288, 0., 298., 0., 298, 298]]).T
        self.xmax = np.array([[self.tT, 3e4, 1e4, 1e4, 400, 400, 400, 400, 400, 1e4, 400, 3e4, 400, 400]]).T
        self.umin = np.array([[-5e2, -2, -2]]).T / self.dt
        self.umax = np.array([[5e2, 2, 2]]).T / self.dt
        self.ymin = self.xmin.take([0, 3]).reshape([-1, 1])
        self.ymax = self.xmax.take([0, 3]).reshape([-1, 1])

        # Basic setup for environment
        self.zero_center_scale = True
        self._set_sym_expressions()
        self.reset()

        self.need_derivs = config['need_derivs']
        self.need_noise_derivs = config['need_noise_derivs']
        self.need_deriv_inverse = config['need_deriv_inverse']

        if self.need_derivs:
            self._eval_model_derivs()

        self.plot_info = {
            'ref_idx_lst': [],
            'state_plot_shape': (3, 5),
            'action_plot_shape': (1, 3),
            'variable_tag_lst': [
                r'Time[hour]', r'$m_{W}[kg]$', r'$m_{A}[kg]$', r'$m_{P}[kg]$',
                r'$T_{R}[K]$', r'$T_{S}[K]$', r'$T_{M}[K]$', r'$T_{EK}[K]$', r'$T_{AWT}[K]$',
                r'$m_F^{acc}[kg]$', r'$T_{adiab}[K]$', r'$\dot{m}_f[kg/h]$',r'$T_{in, M}[K]$', r'$T_{in, EK}[K]$',
                r'$\Delta \dot{m}_f[kg/h]$',r'$\Delta T_{in, M}[K]$', r'$\Delta T_{in, EK}[K]$'
            ]
        }

    def reset(self, x0=None, random_init=False):
        if x0 is None:
            if random_init == False:
                x0 = self.x0
            else:
                x0 = self.x0
                # t, u0 should not be initialized randomly
                x0[1:self.s_dim - 1] = self.descale(np.random.uniform(-0.3, 0.3,
                                                                      [self.s_dim - self.a_dim - 1, 1]),
                                                    self.xmin[1:self.s_dim - 1], self.xmax[1:self.s_dim - 1])

        x0 = self.scale(x0, self.xmin, self.xmax)
        u0 = self.scale(self.u0, self.umin, self.umax)

        self.time_step = 0

        # Parameter uncertainty
        if self.param_uncertainty:
            self.p_sigma = self.param_range * 0.3
            self.p_eps = np.random.normal(size=[self.p_dim, 1])
        else:
            self.p_sigma = self.param_range * 0
            self.p_eps = np.zeros([self.p_dim, 1])

        if self.param_extreme == 'case1':
            self.p_mu = np.array([[950.0 * 0.70, 7.0 * 0.70]]).T
        elif self.param_extreme == 'case2':
            self.p_mu = np.array([[950.0 * 1.30, 7.0 * 1.30]]).T
        else:
            self.p_mu = self.param_real

        return x0, u0

    def ref_traj(self):
        return np.zeros([self.o_dim, ])

    def init_controller(self, o):
        o = self.descale(o, self.ymin, self.ymax)
        t = o[0]

        dm_dot_f = 0.
        if t.item() < 1.1:
            dT_in_M = -1 / self.dt
            dT_in_EK = -1 / self.dt
        else:
            dT_in_M = 5 / self.dt
            dT_in_EK = 5 / self.dt

        u = np.array([dm_dot_f, dT_in_M, dT_in_EK]).reshape([-1, 1])

        return u

    def get_observ(self, state, action):
        observ = self.y_fnc(state, action, self.p_mu, self.p_sigma, self.p_eps).full()

        return observ

    def step(self, state, action):
        self.time_step += 1

        # Scaled state & action
        x = np.clip(state, -1, 2)
        u = action

        # Identify data_type
        is_term = False
        if self.time_step == self.nT:
            is_term = True

        # Integrate ODE
        if not is_term:
            res = self.I_fnc(x0=x,
                             p=np.concatenate([u, self.p_mu, self.p_sigma, np.random.normal(size=[self.p_dim, 1])]))
            xplus = res['xf'].full()
            cost = res['qf'].full()
            derivs = None

            if self.need_derivs:
                _, dfdx, dfdu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                _, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2 = [_.full() for _ in
                                                          self.c_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                d2cdu2_inv, Fc, dFcdx, dFcdu = None, None, None, None

                if self.need_deriv_inverse:
                    U = sp.linalg.cholesky(d2cdu2)  # -Huu_inv @ [Hu, Hux, Hus, Hun]
                    d2cdu2_inv = sp.linalg.solve_triangular(U, sp.linalg.solve_triangular(U.T, np.eye(self.a_dim),
                                                                                          lower=True))

                if self.need_noise_derivs:
                    Fc_derivs = [_.full() for _ in self.Fc_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                    Fc = Fc_derivs[0]
                    dFcdx = np.array(Fc_derivs[1:1 + self.p_dim])
                    dFcdu = np.array(Fc_derivs[1 + self.p_dim:])

                derivs = [dfdx, dfdu, dcdx, dcdu, d2cdx2, d2cdxdu, d2cdu2, d2cdu2_inv, Fc, dFcdx, dFcdu]
        else:
            xplus = x
            cost = self.cT_fnc(x, self.p_mu, self.p_sigma, self.p_eps).full()
            derivs = None

            if self.need_derivs:
                _, dfdx, dfdu = [_.full() for _ in self.dx_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)]
                _, dcTdx, d2cTdx2 = [_.full() for _ in self.cT_derivs(x, self.p_mu, self.p_sigma, self.p_eps)]
                dcTdu = np.zeros([self.a_dim, 1])
                d2cTdxdu = np.zeros([self.s_dim, self.a_dim])
                d2cTdu2 = np.zeros([self.a_dim, self.a_dim])
                d2cTdu2_inv = np.zeros([self.a_dim, self.a_dim])

                Fc, dFcdx, dFcdu = None, None, None

                if self.need_noise_derivs:
                    Fc_derivs = self.Fc_derivs(x, u, self.p_mu, self.p_sigma, self.p_eps)
                    Fc = Fc_derivs[0]
                    dFcdx = np.array(Fc_derivs[1:1 + self.p_dim])
                    dFcdu = np.array(Fc_derivs[1 + self.p_dim:])

                derivs = [dfdx, dfdu, dcTdx, dcTdu, d2cTdx2, d2cTdxdu, d2cTdu2, d2cTdu2_inv, Fc, dFcdx, dFcdu]

        noise = np.zeros_like(xplus)
        state_noise = np.random.normal(np.zeros([self.s_dim - self.a_dim - 1, 1]),
                                       0.005 * np.ones([self.s_dim - self.a_dim - 1, 1]))
        noise[1:self.s_dim - self.a_dim] = state_noise
        xplus = np.clip(xplus + noise, -1, 2)

        return xplus, cost, is_term, derivs

    def system_functions(self, *args):

        x, u, p_mu, p_sigma, p_eps = args

        x = self.descale(x, self.xmin, self.xmax)
        u = self.descale(u, self.umin, self.umax)

        x = ca.fmax(x, self.xmin)
        u = ca.fmin(ca.fmax(u, self.umin), self.umax)

        # if the variables become 2D array, then use torch.mm()
        p = p_mu + p_eps * p_sigma
        t, m_W, m_A, m_P, T_R, T_S, Tout_M, T_EK, Tout_AWT, accum_monom, T_adiab, m_dot_f, T_in_M, T_in_EK = ca.vertsplit(x)
        dm_dot_f, dT_in_M, dT_in_EK = ca.vertsplit(u)
        delH_R, k_0 = ca.vertsplit(p)

        # algebraic equations
        U_m = m_P / (m_A + m_P)
        m_ges = m_W + m_A + m_P
        k_R1 = k_0 * ca.exp(- E_a / (R * T_R)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
        k_R2 = k_0 * ca.exp(- E_a / (R * T_EK)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
        k_K = ((m_W / m_ges) * k_WS) + ((m_A / m_ges) * k_AS) + ((m_P / m_ges) * k_PS)

        dt = 1.
        dm_W = m_dot_f * w_WF
        dm_A = (m_dot_f * w_AF) - (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) - (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
        dm_P = (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
        dT_R = 1./(c_pR * m_ges)   * ((m_dot_f * c_pF * (T_F - T_R)) - (k_K *A_tank* (T_R - T_S)) - (fm_AWT * c_pR * (T_R - T_EK)) + (delH_R * k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))))
        dT_S = 1./(c_pS * m_S)     * ((k_K *A_tank* (T_R - T_S)) - (k_K *A_tank* (T_S - Tout_M)))
        dTout_M = 1./(c_pW * m_M_KW)  * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K *A_tank* (T_S - Tout_M)))
        dT_EK = 1./(c_pR * m_AWT)   * ((fm_AWT * c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT * delH_R))
        dTout_AWT = 1./(c_pW * m_AWT_KW)* ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK)))
        daccum_monom = m_dot_f
        dT_adiab = delH_R/(m_ges*c_pR)*dm_A-(dm_A+dm_W+dm_P)*(m_A*delH_R/(m_ges*m_ges*c_pR))+dT_R


        dx = [dt, dm_W, dm_A, dm_P, dT_R, dT_S, dTout_M, dT_EK, dTout_AWT, daccum_monom, dT_adiab, dm_dot_f, dT_in_M, dT_in_EK]

        dx = ca.vertcat(*dx)
        dx = self.scale(dx, self.xmin, self.xmax, shift=False)

        outputs = ca.vertcat(t, m_P)
        y = self.scale(outputs, self.ymin, self.ymax, shift=True)
        return dx, y

    def cost_functions(self, data_type, *args):
        if data_type == 'path':
            x, u, p_mu, p_sigma, p_eps = args  # scaled variable
        else:  # terminal condition
            x, p_mu, p_sigma, p_eps = args  # scaled variable
            u = np.zeros([self.a_dim, 1])

        Q = np.diag([50])
        R = np.diag([0.002, 0.004, 0.002])
        H = np.array([0.])

        y = self.y_fnc(x, u, p_mu, p_sigma, p_eps)
        t, m_P_denorm = ca.vertsplit(y)

        if data_type == 'path':
            cost = (1 - ca.tanh(m_P_denorm)) @ Q + u.T @ R @ u
        else:  # terminal condition
            cost = (1 - ca.tanh(m_P_denorm)) @ Q

        return cost
