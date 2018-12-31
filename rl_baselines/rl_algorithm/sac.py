import os
import pickle

from stable_baselines import SAC
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines.sac.policies import MlpPolicy, CnnPolicy

from environments.utils import makeEnv
from rl_baselines.base_classes import StableBaselinesRLObject
from rl_baselines.utils import loadRunningAverage, MultiprocessSRLModel


class SACModel(StableBaselinesRLObject):
    """
    Object containing the interface between baselines.sac and this code base
    SAC: Soft Actor Critic
    """
    def __init__(self):
        super(SACModel, self).__init__(name="sac", model_class=SAC)

    def customArguments(self, parser):
        parser.add_argument('--ent-coef', help='The entropy coefficient', type=float, default=0.01)
        parser.add_argument('--batch-size', help='The batch size used for training', type=int, default=64)
        return parser

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

        return loaded_model

    @classmethod
    def makeEnv(cls, args, env_kwargs=None, load_path_normalise=None):
        # Even though DeepQ is single core only, we need to use the pipe system to work
        if env_kwargs is not None and env_kwargs.get("use_srl", False):
            srl_model = MultiprocessSRLModel(1, args.env, env_kwargs)
            env_kwargs["state_dim"] = srl_model.state_dim
            env_kwargs["srl_pipe"] = srl_model.pipe

        env = DummyVecEnv([makeEnv(args.env, args.seed, 0, args.log_dir, env_kwargs=env_kwargs)])

        if args.srl_model != "raw_pixels":
            env = VecNormalize(env, norm_reward=False)
            env = loadRunningAverage(env, load_path_normalise=load_path_normalise)

        return env

    @classmethod
    def getOptParam(cls):
        return {
            # sac param
            "ent_coef": (float, (0.0001, 1)),
            "learning_rate": (float, (0, 0.1)),
            "gradient_steps": (int, (1, 500)),
            "train_freq": (int, (1, 500)),
        }

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        env = self.makeEnv(args, env_kwargs=env_kwargs)

        if train_kwargs is None:
            train_kwargs = {}

        # get the associated policy for the architecture requested
        if args.srl_model == "raw_pixels":
            args.policy = "cnn"
        else:
            args.policy = "mlp"

        self.policy = args.policy
        self.ob_space = env.observation_space
        self.ac_space = env.action_space

        policy_fn = {'cnn': CnnPolicy,
                     'mlp': MlpPolicy}[args.policy]

        param_kwargs = {
            "verbose": 1,
        }

        self.model = self.model_class(policy_fn, env, **{**param_kwargs, **train_kwargs})
        self.model.learn(total_timesteps=args.num_timesteps, seed=args.seed, callback=callback)
        env.close()
