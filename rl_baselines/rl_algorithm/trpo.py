from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack

from rl_baselines.base_classes import StableBaselinesRLObject
from rl_baselines.utils import MultiprocessSRLModel, loadRunningAverage
from environments.utils import makeEnv
from srl_zoo.utils import printYellow


class TRPOModel(StableBaselinesRLObject):
    """
    object containing the interface between baselines.trpo_mpi and this code base
    TRPO: Trust Region Policy Optimization (MPI Implementation)
    """

    LOG_INTERVAL = 10  # log RL model performance every 10 steps
    SAVE_INTERVAL = 1  # Save RL model every 1 steps

    def __init__(self):
        super(TRPOModel, self).__init__(name="trpo", model_class=TRPO)

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
            "vf_stepsize": (float, (1e-2, 1e-5)),
            "entcoeff": (float, (0, 1)),
            "max_kl": (float, (0, 1)),
            "cg_damping": (float, (0, 1)),
            "cg_iters": (int, (1, 10)),
            "vf_iters": (int, (1, 10)),
            "timesteps_per_batch": (int, (32, 2048))
        }

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        if train_kwargs is None:
            train_kwargs = {}

        if args.srl_model == "raw_pixels":
            printYellow("Warning: TRPO can have memory issues when running with raw_pixels")

        param_kwargs = {
            "verbose": 1,
            "timesteps_per_batch": 128,
            "entcoeff": 0.01,
            "vf_stepsize": 2.5e-4,
            "gamma": 0.99,
            "lam": 0.95,
            "cg_iters": 4,
            "cg_damping": 0.2,
            "vf_iters": 3,
            "max_kl": 0.01,
        }

        super().train(args, callback, env_kwargs, {**param_kwargs, **train_kwargs})
