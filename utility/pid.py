import numpy as np


class PID(object):
    """PID controller for generating initial warm-up data"""
    def __init__(self):
        self._a_dim = None
        self._o_dim = None
        self._dt = None

        self._Kp = None
        self._Ki = None
        self._Kd = None
        
        self._ref = None

    def reset(self):
        self._integral = np.zeros((self._o_dim, 1))
        self._prev_error = np.zeros((self._o_dim, 1))

    @property
    def get_info(self):
        return {'o_dim': self._o_dim, 'a_dim': self._a_dim, 'dt': self._dt}
    
    @get_info.setter
    def set_info(self, info):
        self._a_dim = info['a_dim']
        self._o_dim = info['o_dim']
        self._dt = info['dt']

    @property
    def get_gain(self):
        return {'Kp': self._Kp, 'Ki': self._Ki, 'Kd': self._Kd}

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
        self._ref = ref.reshape(self._o_dim, 1)

    def ctrl(self, observ):
        error = self._ref - observ

        proportional = error
        integral = self._integral + error * self._dt
        derivative = (error - self._prev_error) / self._dt
        
        action = self._Kp @ proportional + self._Ki @ integral + self._Kd @ derivative

        self._integral = integral
        self._prev_error = error

        return action
