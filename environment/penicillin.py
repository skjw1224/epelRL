import os
import numpy as np
import casadi as ca
import scipy as sp
import matplotlib.pyplot as plt

from .base_environment import Environment

# Physio-chemical parameters for the penicillin fed-batch reactor
# Biomass related
mu_x = 0.092    # Maximum specific growth rate [1/h]
Kx = 0.15       # Contois saturation constant [g/L]
Kox = 2e-2      # Oxygen limitation constant for biomass [-]
kg = 7e3        # Arrhenius constant for cell growth [K]
kd = 1e33       # Arrhenius constant for cell death [K]
Eg = 5100       # Activation energy for cell growth [cal/mol]
Ed = 50000      # Activation energy for cell death [cal/mol]
Yxs = 0.45      # Yield (biomass/glucose) [g/g]
Yxo = 0.04      # Yield (biomass/oxygen) [g/g]

# pH related
K1 = 1E-10      # pH dependency constant [mol/L]
K2 = 7E-5       # pH dependency constant [mol/L]
gamma = 1E-5    # Empirical proportionality constant [mol/g]
Cab = 3         # Acid and base flow concentration [M]

# Penicillin related
mup = 0.005     # Specific rate of penicillin prod. [1/h]
Kp = 0.0002     # Inhibition constant [g/L]
KI = 0.10       # Dissociation constant [g/L]
Kop = 5e-4      # Oxygen limitation constant for penicillin [-]
p = 3           # Dissolved oxygen dependency constant [-].
Yps = 0.90      # Yield (penicillin/glucose) [g/g]
Ypo = 0.20      # Yield (penicillin/oxygen) [g/g]
K = 0.04        # Penicillin hydrolysis rate constant [1/h]

# Substrate related
mx = 0.014      # Maintenance coefficient on substrate [1/h]
mo = 0.467      # Maintenance coefficient on oxygen [1/h]
Clstar = 1.16   # Saturation concentration of CL [g/L]
alpha = 70      # Mass transfer modeling constant [-]
beta = 0.4      # Mass transfer modeling constant [-]
Pw = 29.9       # Agitation power [kW]
fg = 0.1        # Oxygen flow rate [g/L]

# Volume related
lambda_par = 2.5e-4 # Volume loss constant [1/h]
T0 = 273.15     # Freezing temperature [K]
Tv = 373.15     # Boiling temperature [K]

# Heat related
rq1 = 60        # Yield coefficient of heat generation [cal/g]
rq2 = 1.6783e-4 # Maintenance coefficient of heat generation [cal/g/h]
rhocp = 1500    # Volumetric heat capacity of medium liquid [cal/K]
rhoccpc = 1000  # Volumetric heat capacity of cooling liquid [cal/K]
a = 1000        # Heat transfer coefficient of cooling liquid [cal/K]
b = 0.60        # Empirical heat generation constant [-]
Th = 313.15     # Heating water temperature [K]
Tc = 283.15     # Cooling water temperature [K]
q = 0           # Ratio of heating and cooling water flow rate

# CO2 related
alpha1 = 0.143  # Empirical constant [mmol/g]
alpha2 = 4e-7   # Empirical constant [mmol/g/h]
alpha3 = 1e-4   # Empirical constant [mmol/L/h]

# Initial MVs
sf = 600        # Biomass concentration [g/L]
F = 0.05        # Substrate feed flow rate [L/h]
Tf = 298        # Feed temperature [K]
Fa = 3e-5       # Acid flow rate [L/h]
Fb = 1.5e-4     # Base flow rate [L/h]
Fc = 2.e3       # Cooling water flow rate [L/h]
R = 1.987       # Gas constant [cal/K/mol]
H = 10**-5.1        # Setpoint of H concentration [M]
T = 297.        # Setpoint of temperature [K]

# Parameters with uncertainty
Kx = 0.15       # Contois saturation constant [g/L]
class PENICILLIN(Environment):
    def __init__(self, config):
        self.env_name = 'PENICILLIN'
        self.config = config

        # Uncertain parameters: Kx
        self.param_real = np.array([[Kx]]).T
        self.param_range = np.array([[Kx * 0.3]]).T
        self.p_mu = self.param_real
        self.p_sigma = np.zeros([np.shape(self.param_real)[0], 1])
        self.p_eps = np.zeros([np.shape(self.param_real)[0], 1])
        self.param_uncertainty = False
        self.param_extreme = False

        # Dimension
        self.s_dim = 8
        self.a_dim = 1
        self.o_dim = 8
        self.p_dim = np.shape(self.param_real)[0]

        self.t0 = 0.
        self.dt = 1. # [h]
        self.tT = 400.

        # Full version
        # state: t, X, S, Cl, P, V, CO2, Q, H, T, Sa, F
        # action: F, Fa, Fb, Fc, fg, Pw, q

        # Simple version: Fixed pH, Temp, Single MV
        # state: t, X, S, Cl, P, V, Sa, F
        # action: dF
        # observation: t, X, S, Cl, P, V, Sa, F

        self.x0 = np.array([[self.t0, 0.1, 15., 1.16, 0., 100., 1500, 0.01]]).T
        self.u0 = np.array([[0.05]]).T
        self.nT = int(self.tT / self.dt)  # episode length

        self.xmin = np.array([[self.t0, 0., 0., 0., 0., 80, 1500, 0.]]).T
        self.xmax = np.array([[self.tT, 50., 25., 1.5, 4., 110, 15000, 0.1]]).T
        self.umin = np.array([[-0.01]]).T
        self.umax = np.array([[0.01]]).T
        self.ymin = self.xmin
        self.ymax = self.xmax
        self.emin = np.array([[0., 0.]]).T
        self.emax = np.array([[250., 1.]]).T

        # Basic setup for environment
        self.zero_center_scale = True
        self._set_sym_expressions()
        self.reset()

        self.need_derivs = config.need_derivs
        self.need_noise_derivs = config.need_noise_derivs
        self.need_deriv_inverse = config.need_deriv_inverse

        if self.need_derivs:
            self._eval_model_derivs()

        self.plot_info = {
            'ref_idx_lst': [],
            'state_plot_shape': (2, 4),
            'action_plot_shape': (1, 1),
            'variable_tag_lst': [
                r'Time[hour]', r'$X[g/L]$', r'$S[g/L$]', r'$C_L[g/L]$',
                r'$P[g/L]$', r'$V[L]$', r'$S_{accum}[g/L]$', r'$F[L/h]$',
                r'$dF[L/h]$'
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
            self.p_mu = np.array([[Kx * 0.70]]).T
        elif self.param_extreme == 'case2':
            self.p_mu = np.array([[Kx * 1.30]]).T
        else:
            self.p_mu = self.param_real

        return x0, u0

    def ref_traj(self):
        return np.zeros([self.o_dim, ])

    def init_controller(self, o, amplitude=0.04):
        o = self.descale(o, self.ymin, self.ymax)
        t = o[0]
        F = o[-1]

        # u = self.umax
        # print(t, F)

        if t.item() >= 50 and F < amplitude:
            u = self.umax
        else:
            u = 0.
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
        t, X, S, Cl, P, V, Sa, F = ca.vertsplit(x)
        dFval = ca.vertsplit(u)[0]
        Kx = ca.vertsplit(p)[0]

        # algebraic equations
        mux_mod = mu_x / (1 + K1 / H + H / K2)
        kinetics = kg * ca.exp(-Eg / (R * T)) - kd * ca.exp(-Ed / (R * T))

        mu = mux_mod * S / (Kx * X + S) * Cl / (Kox * X + Cl) * kinetics
        mupp = mup * S / (Kp + S + S ** 2 / KI) * Cl ** p / (Kop * X + Cl ** p)
        B = ((1e-14 / H - H) * V + (-Cab * Fa + Fb) * self.dt) / (V + (Fa + Fb) * self.dt)
        Kla = alpha * ca.sqrt(fg) * (Pw / V) ** beta
        Floss = V * lambda_par * (ca.exp(5 * (T - T0) / (Tv - T0)) - 1)

        dt = 1.
        dV = F + Fa + Fb - Floss
        dX = mu * X - X / V * dV
        dS = -mu / Yxs * X - mupp / Yps * X - mx * X + F * sf / V - S / V * dV
        dCl = -mu / Yxo * X - mupp / Ypo * X - mo * X + Kla * (Clstar - Cl) - Cl / V * dV
        dP = mupp * X - K * P - P / V * dV
        dSa = sf * F
        dF = dFval

        # Full version
        # dQdt = rq1 * dXdt * V + rq2 * X * V
        # dCO2dt = alpha1 * dXdt + alpha2 * X + alpha3
        # dHdt = gamma * (mu * X - F * X / V) + ((-B + ca.sqrt(B ** 2 + 4e-14)) / 2 - H) / delt
        # dTdt = F / V * (Tf - T) + 1 / (V * rhocp) * (Q - a * Fc ** b / (1 + a * Fc ** (b - 1) / (2 * rhoccpc))) * (
        #         T - q * Th - (1 - q) * Tc)

        dx = [dt, dX, dS, dCl, dP, dV, dSa, dF]

        dx = ca.vertcat(*dx)
        dx = self.scale(dx, self.xmin, self.xmax, shift=False)

        outputs = x
        y = self.scale(outputs, self.ymin, self.ymax, shift=True)
        return dx, y

    def cost_functions(self, data_type, *args):
        if data_type == 'path':
            x, u, p_mu, p_sigma, p_eps = args  # scaled variable
        else:  # terminal condition
            x, p_mu, p_sigma, p_eps = args  # scaled variable
            u = np.zeros([self.a_dim, 1])

        Qc = np.diag([1.]) * 0.001
        Hc = np.array([[0.1, 2.]])

        y = self.y_fnc(x, u, p_mu, p_sigma, p_eps)
        y = self.descale(y, self.ymin, self.ymax)
        t, X, S, Cl, P, V, Sa, F = ca.vertsplit(y)
        prod = P * V
        yield_ = P * V / Sa
        econ = self.scale(ca.vertcat(prod, yield_), self.emin, self.emax)
        ecost = 50 * (1 - ca.tanh(Hc @ econ))
        ucost = u.T @ Qc @ u

        if data_type == 'path':
            cost = ecost * 0.01 + ucost
        else:  # terminal condition
            cost = ecost

        return cost

