import os
import pickle

from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy, MlpLstmPolicy, \
    MlpLnLstmPolicy

from rl_baselines.utils import createEnvs


class BaseRLObject:
    """
    Base object for RL algorithms
    """

    # if callback frequency needs to be changed, overwrite this.
    LOG_INTERVAL = 100  # log RL model performance every 100 steps
    SAVE_INTERVAL = 200  # Save RL model every 200 steps

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
    def getOptParam(cls):
        return None

    @classmethod
    def parserHyperParam(cls, hyperparam):
        """
        parses the hyperparameters into the expected type

        :param hyperparam: (dict) the input hyperparameters (can be None)
        :return: (dict) the parsed hyperparam dict (returns at least an empty dict)
        """
        opt_param = cls.getOptParam()
        parsed_hyperparam = {}

        if opt_param is not None and hyperparam is not None:
            for name, val in hyperparam.items():
                if name not in opt_param:
                    raise AssertionError("Error: hyperparameter {} not in list of valid hyperparameters".format(name))
                if isinstance(opt_param[name][0], tuple):
                    parsed_hyperparam[name] = opt_param[name][0][1](val)
                else:
                    parsed_hyperparam[name] = opt_param[name][0](val)

        return parsed_hyperparam

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

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        """
        Makes an environment and trains the model on it
        :param args: (argparse.Namespace Object)
        :param callback: (function)
        :param env_kwargs: (dict) The extra arguments for the environment
        :param train_kwargs: (dict) The list of all training agruments (used in hyperparameter search)
        """
        raise NotImplementedError()


class StableBaselinesRLObject(BaseRLObject):
    """
    Base object for the Stable Baselines RL algorithms
    """

    def __init__(self, name, model_class):
        super(StableBaselinesRLObject, self).__init__()
        self.name = name
        self.model_class = model_class
        self.model = None
        self.states = None
        self.ob_space = None
        self.ac_space = None
        self.policy = None
        self.load_rl_model_path = None

    def save(self, save_path, _locals=None):
        """
        Save the model to a path
        :param save_path: (str)
        :param _locals: (dict) local variable from callback, if present
        """
        assert self.model is not None, "Error: must train or load model before use"
        model_save_name = self.name + ".pkl"
        if os.path.basename(save_path) == model_save_name:
            model_save_name = self.name + "_model.pkl"

        self.model.save(os.path.dirname(save_path) + "/" + model_save_name)
        save_param = {
            "ob_space": self.ob_space,
            "ac_space": self.ac_space,
            "policy": self.policy
        }
        with open(save_path, "wb") as f:
            pickle.dump(save_param, f)

    def setLoadPath(self, load_path):
        """
        Set the path to later load the parameters of a trained rl model
        :param load_path: (str)
        :return: None
        """
        self.load_rl_model_path = load_path
       
    @classmethod
    def load(cls, load_path, args=None):
        """
        Load the model from a path
        :param load_path: (str)
        :param args: (dict) the arguments used
        :return: (BaseRLObject)
        """
        with open(load_path, "rb") as f:
            save_param = pickle.load(f)

        loaded_model = cls()
        loaded_model.__dict__ = {**loaded_model.__dict__, **save_param}

        model_save_name = loaded_model.name + ".pkl"
        if os.path.basename(load_path) == model_save_name:
            model_save_name = loaded_model.name + "_model.pkl"

        loaded_model.model = loaded_model.model_class.load(os.path.dirname(load_path) + "/" + model_save_name)
        loaded_model.states = loaded_model.model.initial_state

        return loaded_model

    def customArguments(self, parser):
        """
        Added arguments for training
        :param parser: (ArgumentParser Object)
        :return: (ArgumentParser Object)
        """
        parser.add_argument('--policy', help='Policy architecture', choices=['feedforward', 'lstm', 'lnlstm'],
                            default='feedforward')
        return parser

    def getActionProba(self, observation, dones=None):
        """
        From an observation returns the associated action probability
        :param observation: (numpy float)
        :param dones: ([bool])
        :return: (numpy float)
        """
        assert self.model is not None, "Error: must train or load model before use"
        return self.model.action_probability(observation, self.states, dones)

    def getAction(self, observation, dones=None):
        """
        From an observation returns the associated action
        :param observation: (numpy float)
        :param dones: ([bool])
        :return: (numpy float)
        """
        assert self.model is not None, "Error: must train or load model before use"
        actions, self.states = self.model.predict(observation, self.states, dones)
        return actions

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

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        """
        Makes an environment and trains the model on it
        :param args: (argparse.Namespace Object)
        :param callback: (function)
        :param env_kwargs: (dict) The extra arguments for the environment
        :param train_kwargs: (dict) The list of all training agruments (used in hyperparameter search)
        """
        envs = self.makeEnv(args, env_kwargs=env_kwargs)

        if train_kwargs is None:
            train_kwargs = {}

        # get the associated policy for the architecture requested
        if args.srl_model == "raw_pixels":
            if args.policy == "feedforward":
                args.policy = "cnn"
            else:
                args.policy = "cnn-" + args.policy
        else:
            if args.policy == "feedforward":
                args.policy = "mlp"

        self.policy = args.policy
        self.ob_space = envs.observation_space
        self.ac_space = envs.action_space

        policy_fn = {'cnn': "CnnPolicy",
                     'cnn-lstm': "CnnLstmPolicy",
                     'cnn-lnlstm': "CnnLnLstmPolicy",
                     'mlp': "MlpPolicy",
                     'lstm': "MlpLstmPolicy",
                     'lnlstm': "MlpLnLstmPolicy"}[args.policy]
        if self.load_rl_model_path is not None:
            print("Load trained model from the path: ", self.load_rl_model_path)
            self.model = self.model_class.load(self.load_rl_model_path, envs, **train_kwargs)
        else:
            self.model = self.model_class(policy_fn, envs, **train_kwargs)
        self.model.learn(total_timesteps=args.num_timesteps, seed=args.seed, callback=callback)
        envs.close()
