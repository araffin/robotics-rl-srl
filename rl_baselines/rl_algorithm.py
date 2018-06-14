from rl_baselines.utils import createEnvs


class BaseRLObject:
    """
    Base object for RL algorithms
    """
    def __init__(self):
        pass

    def save(self, save_path, _locals=None):
        """
        Save the model to a path
        :param save_path: (str)
        :param _locals: (dict) local variable from callback, if present
        """
        raise NotImplementedError()

    @classmethod
    def load(cls, load_path, args=None):
        """
        Load the model from a path
        :param load_path: (str)
        :param args: (dict) the arguments used
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

    def getAction(self, observation, dones=None):
        """
        From an observation returns the associated action
        :param observation: (numpy float)
        :param dones: ([bool])
        :return: (numpy float)
        """
        raise NotImplementedError()

    @classmethod
    def makeEnv(cls, args, env_kwargs=None, load_path_normalise=None):
        """
        Makes an environment and returns it
        :param args: (argparse.Namespace Object)
        :param env_kwargs: (dict) The extra arguments for the environment
        :param load_path_normalise: (str) the path to loading the rolling average, None if not available or wanted.
        :return: (Gym env)
        """
        return createEnvs(args, env_kwargs=env_kwargs, load_path_normalise=load_path_normalise)

    def train(self, args, callback, env_kwargs=None):
        """
        Makes an environment and trains the model on it
        :param args: (argparse.Namespace Object)
        :param callback: (function)
        :param env_kwargs: (dict) The extra arguments for the environment
        """
        raise NotImplementedError()
