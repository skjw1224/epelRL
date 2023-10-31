import abc


class Environment(object, metaclass=abc.ABCMeta):
    """
    Base class for environment
    """

    @abc.abstractmethod
    def reset(self):
        """
        Reset environment into initial state
        """
        pass

    @abc.abstractmethod
    def step(self, time, state, action):
        """
        Compute next state from current state and action
        """
        pass

    @abc.abstractmethod
    def system_functions(self):
        """
        Equations that describe dynamics of the system
        """
        pass

    @abc.abstractmethod
    def cost_functions(self, data_type):
        """
        Compute the cost (reward) of current state and action
        """
        pass

    @abc.abstractmethod
    def plot_trajectory(self, trajectory, episode, controller, path):
        """
        Plot the control trajectories
        """
        pass

