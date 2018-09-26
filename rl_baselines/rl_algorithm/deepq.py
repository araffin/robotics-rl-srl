import pickle
import os

from stable_baselines import DQN
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv

from environments.utils import makeEnv
from rl_baselines.base_classes import StableBaselinesRLObject
from rl_baselines.utils import loadRunningAverage, MultiprocessSRLModel


class DQNModel(StableBaselinesRLObject):
    """
    object containing the interface between baselines.deepq and this code base
    DQN: https://arxiv.org/pdf/1312.5602v1.pdf
    """
    def __init__(self):
        super(DQNModel, self).__init__(name="deepq", model_class=DQN)

    def customArguments(self, parser):
        parser.add_argument('--prioritized', type=int, default=1)
        parser.add_argument('--dueling', type=int, default=1)
        parser.add_argument('--buffer-size', type=int, default=int(1e3), help="Replay buffer size")
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
        # Even though DQN is single core only, we need to use the pipe system to work
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
            "learning_rate": (float, (0, 0.1)),
            "exploration_fraction": (float, (0, 1)),
            "exploration_final_eps": (float, (0, 1)),
            "train_freq": (int, (1, 10)),
            "learning_starts": (int, (10, 10000)),
            "target_network_update_freq": (int, (10, 10000)),
            "gamma": (float, (0, 1)),
            "batch_size": (int, (8, 128)),
        }

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        env = self.makeEnv(args, env_kwargs=env_kwargs)

        if train_kwargs is None:
            train_kwargs = {}

        if args.srl_model != "raw_pixels":
            args.policy = "mlp"
            policy_fn = 'MlpPolicy'
            # policy_fn = deepq_models.mlp([64, 64])
        else:
            # Atari CNN
            args.policy = "cnn"
            policy_fn = 'CnnPolicy'
            # policy_fn = deepq_models.cnn_to_mlp(
            #     convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            #     hiddens=[256],
            #     dueling=bool(args.dueling),
            # )

        self.policy = args.policy
        self.ob_space = env.observation_space
        self.ac_space = env.action_space

        param_kwargs = {
            "verbose": 1,
            "learning_rate": 1e-4,
            "buffer_size": args.buffer_size,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.01,
            "train_freq": 4,
            "learning_starts": 500,
            "target_network_update_freq": 500,
            "gamma": 0.99,
            "prioritized_replay": bool(args.prioritized),
            "prioritized_replay_alpha": 0.6
        }

        self.model = self.model_class(policy_fn, env, **{**param_kwargs, **train_kwargs})
        self.model.learn(total_timesteps=args.num_timesteps, seed=args.seed, callback=callback)
        env.close()
