from stable_baselines.ppo1 import PPO1
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack

from rl_baselines.base_classes import StableBaselinesRLObject
from rl_baselines.utils import MultiprocessSRLModel, loadRunningAverage
from environments.utils import makeEnv
from srl_zoo.utils import printYellow


class PPO1Model(StableBaselinesRLObject):
    """
    object containing the interface between baselines.ppo1 and this code base
    PPO1: Proximal Policy Optimization (MPI Implementation)
    """

    LOG_INTERVAL = 10  # log RL model performance every 10 steps
    SAVE_INTERVAL = 1  # Save RL model every 1 steps

    def __init__(self):
        super(PPO1Model, self).__init__(name="ppo1", model_class=PPO1)

    def customArguments(self, parser):
        super().customArguments(parser)
        parser.add_argument('--lr-schedule', help='Learning rate schedule', default='constant',
                            choices=['linear', 'constant'])
        return parser

    @classmethod
    def makeEnv(cls, args, env_kwargs=None, load_path_normalise=None):
        # Even though DeepQ is single core only, we need to use the pipe system to work
        if env_kwargs is not None and env_kwargs.get("use_srl", False):
            srl_model = MultiprocessSRLModel(1, args.env, env_kwargs)
            env_kwargs["state_dim"] = srl_model.state_dim
            env_kwargs["srl_pipe"] = srl_model.pipe

        envs = DummyVecEnv([makeEnv(args.env, args.seed, 0, args.log_dir, env_kwargs=env_kwargs)])
        envs = VecFrameStack(envs, args.num_stack)

        if args.srl_model != "raw_pixels":
            printYellow("Using MLP policy because working on state representation")
            envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
            envs = loadRunningAverage(envs, load_path_normalise=load_path_normalise)

        return envs

    @classmethod
    def getOptParam(cls):
        return {
            "lam": (float, (0, 1)),
            "gamma": (float, (0, 1)),
            "optim_stepsize": (float, (1e-2, 1e-5)),
            "entcoeff": (float, (0, 1)),
            "clip_param": (float, (0, 1)),
            "optim_epochs": (int, (1, 10)),
            "timesteps_per_actorbatch": (int, (32, 2048)),
            "schedule": ((list, str), ['linear', 'constant'])
        }

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        if train_kwargs is None:
            train_kwargs = {}

        param_kwargs = {
            "verbose": 1,
            "timesteps_per_actorbatch": 128,
            "entcoeff": 0.01,
            "optim_stepsize": 2.5e-4,
            "gamma": 0.99,
            "lam": 0.95,
            "optim_epochs": 4,
            "clip_param": 0.2,
            "schedule": args.lr_schedule
        }

        super().train(args, callback, env_kwargs, {**param_kwargs, **train_kwargs})
