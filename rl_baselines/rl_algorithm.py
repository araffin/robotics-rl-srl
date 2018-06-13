class BaseRLObject:
    """
    Base object for RL algorithms
    """
    def __init__(self):
        pass

    def save(self, save_path):
        """
        Save the model to a path
        :param save_path: (str)
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, save_path):
        """
        Load the model from a path
        :param save_path: (str)
        :return: (BaseRLObject)
        """
        raise NotImplementedError()

    def customArguments(self, parser):
        """
        Added arguments for training
        :param parser: (ArgumentParser Object)
        :return: (ArgumentParser Object)
        """
        raise NotImplementedError()

    def getAction(self, observation):
        """
        From an observation returns the associated action
        :param observation: (numpy float)
        :return: (numpy float)
        """
        raise NotImplementedError()

    def train(self, args, callback, env_kwargs=None):
        """
        Makes an environment and trains the model on it
        :param args: (argparse.Namespace Object)
        :param callback: (function)
        :param env_kwargs: (dict) The extra arguments for the environment
        """
        raise NotImplementedError()
