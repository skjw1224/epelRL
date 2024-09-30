import abc


class Algorithm(object, metaclass=abc.ABCMeta):
    """
    Base class for RL algorithms
    """

    @abc.abstractmethod
    def ctrl(self, state):
        """
        Return control actions computed from the policy or the initial controller
        """
        pass

    @abc.abstractmethod
    def train(self):
        """
        Train agent
        """
        pass

    @abc.abstractmethod
    def warm_up_train(self):
        """
        Train agent with warm-up policy
        """
        pass

    @abc.abstractmethod
    def add_experience(self, experience):
        """
        Add experience to the replay buffer
        """
        pass

    @abc.abstractmethod
    def save(self, path, file_name):
        """
        Save the model to the path with file_name
        """
        pass

    @abc.abstractmethod
    def load(self, path, file_name):
        """
        Load the model from the path with file_name
        """
        pass
