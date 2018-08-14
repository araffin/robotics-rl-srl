from stable_baselines.deepq import DeepQ
from stable_baselines.deepq import models as deepq_models
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from rl_baselines.base_classes import StableBaselinesRLObject
from environments.utils import makeEnv
from rl_baselines.utils import WrapFrameStack, loadRunningAverage, MultiprocessSRLModel


class DeepQModel(StableBaselinesRLObject):
    """
    object containing the interface between baselines.deepq and this code base
    DeepQ: https://arxiv.org/pdf/1312.5602v1.pdf
    """
    def __init__(self):
        super(DeepQModel, self).__init__(name="deepq", model_class=DeepQ)

    def customArguments(self, parser):
        parser.add_argument('--prioritized', type=int, default=1)
        parser.add_argument('--dueling', type=int, default=1)
        parser.add_argument('--buffer-size', type=int, default=int(1e3), help="Replay buffer size")
        return parser

    @classmethod
    def makeEnv(cls, args, env_kwargs=None, load_path_normalise=None):
        # Even though DeepQ is single core only, we need to use the pipe system to work
        if env_kwargs is not None and env_kwargs.get("use_srl", False):
            srl_model = MultiprocessSRLModel(1, args.env, env_kwargs)
            env_kwargs["state_dim"] = srl_model.state_dim
            env_kwargs["srl_pipe"] = srl_model.pipe

        env = DummyVecEnv([makeEnv(args.env, args.seed, 0, args.log_dir, env_kwargs=env_kwargs)])

        if args.srl_model != "raw_pixels":
            env = VecNormalize(env)
            env = loadRunningAverage(env, load_path_normalise=load_path_normalise)

        # Normalize only raw pixels
        # WARNING: when using framestacking, the memory used by the replay buffer can grow quickly
        return WrapFrameStack(env, args.num_stack, normalize=args.srl_model == "raw_pixels")

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        env = self.makeEnv(args, env_kwargs=env_kwargs)

        if train_kwargs is None:
            train_kwargs = {}

        if args.srl_model != "raw_pixels":
            args.policy = "mlp"
            policy_fn = deepq_models.mlp([64, 64])
        else:
            # Atari CNN
            args.policy = "cnn"
            policy_fn = deepq_models.cnn_to_mlp(
                convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                hiddens=[256],
                dueling=bool(args.dueling),
            )

        self.policy = args.policy
        self.ob_space = env.observation_space
        self.ac_space = env.action_space

        param_kwargs = {
            "verbose": 1,
            "learning_rate": 1e-4,
            "max_timesteps": args.num_timesteps,
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

        self.model = self.model_class(policy_fn, env, {**param_kwargs, **train_kwargs})
        self.model.learn(total_timesteps=args.num_timesteps, seed=args.seed, callback=callback)
        env.close()
