import numpy as np

class InitCtrl(object):
    """Custom initial controller for generating initial warm-up data"""

    def __init__(self):
        self._a_dim = None
        self._o_dim = None
        self._dt = None

        self._controller = None

    def reset(self):
        pass

    @property
    def get_info(self):
        return {'o_dim': self._o_dim, 'a_dim': self._a_dim, 'dt': self._dt}

    @get_info.setter
    def set_info(self, info):
        self._a_dim = info['a_dim']
        self._o_dim = info['o_dim']
        self._dt = info['dt']

    @property
    def get_controller(self):
        return self._controller

    @get_controller.setter
    def set_controller(self, controller):
        self._controller = controller
    def ctrl(self, observ):
        action = self._controller(observ)
        return action
