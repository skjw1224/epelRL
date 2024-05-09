import numpy as np


class PID(object):
    def __init__(self, pid_info):
        self.a_dim = pid_info['a_dim']
        self.o_dim = pid_info['o_dim']
        self.dt = pid_info['dt']

        self._Kp = np.zeros((self.a_dim, self.o_dim))
        self._Ki = np.zeros((self.a_dim, self.o_dim))
        self._Kd = np.zeros((self.a_dim, self.o_dim))
        
        self._ref = np.zeros((self.o_dim, 1))

    def reset(self):
        self.integral = np.zeros((self.o_dim, 1))
    
    @property
    def get_gain(self):
        gain = {
            'Kp': self._Kp,
            'Ki': self._Ki,
            'Kd': self._Kd,
        }
        return gain

    @get_gain.setter
    def set_gain(self, gain):
        self._Kp = gain['Kp']
        self._Ki = gain['Ki']
        self._Kd = gain['Kd']

    @property
    def get_reference(self):
        return self._ref

    @get_reference.setter
    def set_reference(self, ref):
        self._ref = ref

    def ctrl(self, observ):
        error = self._ref - observ

        proportional = error
        integral = self.integral + error * self.dt
        derivative = (error - self.prev_error) / self.dt
        
        action = self._Kp @ proportional + self._Ki @ integral + self._Kd @ derivative

        self.integral = integral
        self.prev_error = error

        return action
