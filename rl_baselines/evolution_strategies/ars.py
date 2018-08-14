import time
import pickle

import numpy as np
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_frame_stack import VecFrameStack
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from rl_baselines.base_classes import BaseRLObject
from environments import ThreadingType
from environments.registry import registered_env
from environments.utils import makeEnv
from rl_baselines.utils import loadRunningAverage, MultiprocessSRLModel, softmax
from srl_zoo.utils import printYellow


class ARSModel(BaseRLObject):
    """
    Implementation of Augmented Random Search algorithm for gym environment
    https://arxiv.org/abs/1803.07055
    """
    def __init__(self):
        super(ARSModel, self).__init__()
        self.n_population = None
        self.top_population = None  # how many of the population are used in updating
        self.step_size = None  # the step size for the parameter update
        self.exploration_noise = None  # standard deviation of the exploration noise
        self.continuous_actions = None
        self.max_step_amplitude = None  # the maximum amplitude factor for step_size
        self.deterministic = None
        self.M = None  # The linear policy, initialized to zero

    def save(self, save_path, _locals=None):
        assert self.M is not None, "Error: must train or load model before use"
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, load_path, args=None):
        with open(load_path, "rb") as f:
            class_dict = pickle.load(f)
        loaded_model = ARSModel()
        loaded_model.__dict__ = class_dict
        return loaded_model

    def customArguments(self, parser):
        parser.add_argument('--num-population', help='Number of population (each one has 2 threads)', type=int,
                            default=10)
        parser.add_argument('--exploration-noise', help='The standard deviation of the exploration noise', type=float,
                            default=0.02)
        parser.add_argument('--step-size', help='The step size for param update', type=float, default=0.02)
        parser.add_argument('--top-population', help='Number of top population to use in update', type=int, default=2)
        parser.add_argument('--algo-type', help='"v1" is standard ARS, "v2" is for rolling average normalization.',
                            type=str, default="v2", choices=["v1", "v2"])
        parser.add_argument('--max-step-amplitude', type=float, default=10,
                            help='Set the maximum update vectors amplitude (mesured in factors of step_size)')
        parser.add_argument('--deterministic', action='store_true', default=False,
                            help='do a deterministic approach for the actions on the output of the policy')
        return parser

    def getActionProba(self, observation, dones=None, delta=0):
        """
        returns the action probability distribution, from a given observation.
        :param observation: (numpy int or numpy float)
        :param dones: ([bool])
        :param delta: (numpy float or float) The exploration noise applied to the policy, set to 0 for no noise.
        :return: (numpy float)
        """
        assert self.M is not None, "Error: must train or load model before use"
        action = np.dot(observation, self.M + delta)
        if self.continuous_actions:
            return action
        else:
            return softmax(action)

    def getAction(self, observation, dones=None, delta=0):
        """
        From an observation returns the associated action
        :param observation: (numpy int or numpy float)
        :param dones: ([bool])
        :param delta: (numpy float or float) The exploration noise applied to the policy, set to 0 for no noise.
        :return: (numpy float)
        """
        assert self.M is not None, "Error: must train or load model before use"
        action = np.dot(observation, self.M + delta)

        if not self.continuous_actions:
            if self.deterministic:
                action = np.argmax(action, axis=1)
            else:
                action = np.array([np.random.choice(len(a), p=a) for a in softmax(action)])

        return action

    @classmethod
    def getOptParam(cls):
        return {
            "top_population": (int, (1, 5)),
            "exploration_noise": (float, (0, 0.1)),
            "step_size": (float, (0, 0.1)),
            "max_step_amplitude": (float, (1, 100))
        }

    @classmethod
    def makeEnv(cls, args, env_kwargs=None, load_path_normalise=None):
        if "num_population" in args.__dict__:
            args.num_cpu = args.num_population * 2

        assert not (registered_env[args.env][3] is ThreadingType.NONE and args.num_cpu != 1), \
            "Error: cannot have more than 1 CPU for the environment {}".format(args.env)
        if env_kwargs is not None and env_kwargs.get("use_srl", False):
            srl_model = MultiprocessSRLModel(args.num_cpu, args.env, env_kwargs)
            env_kwargs["state_dim"] = srl_model.state_dim
            env_kwargs["srl_pipe"] = srl_model.pipe

        envs = [makeEnv(args.env, args.seed, i, args.log_dir, allow_early_resets=True, env_kwargs=env_kwargs)
                for i in range(args.num_cpu)]
        envs = SubprocVecEnv(envs)
        envs = VecFrameStack(envs, args.num_stack)
        if args.srl_model != "raw_pixels" and args.algo_type == "v2":
            envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
            envs = loadRunningAverage(envs, load_path_normalise=load_path_normalise)
        return envs

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        assert args.top_population <= args.num_population, \
            "Cannot select top %d, from population of %d." % (args.top_population, args.num_population)
        assert args.num_population > 1, "The population cannot be less than 2."

        env = self.makeEnv(args, env_kwargs)

        # set hyperparameters
        args.__dict__.update(train_kwargs)

        if args.continuous_actions:
            action_space = np.prod(env.action_space.shape)
        else:
            action_space = env.action_space.n

        self.M = np.zeros((np.prod(env.observation_space.shape), action_space))
        self.n_population = args.num_population
        self.top_population = args.top_population
        self.step_size = args.step_size
        self.exploration_noise = args.exploration_noise
        self.continuous_actions = args.continuous_actions
        self.max_step_amplitude = args.max_step_amplitude
        self.deterministic = args.deterministic
        num_updates = (int(args.num_timesteps) // args.num_population * 2)

        start_time = time.time()
        step = 0
        while step < num_updates:
            r = np.zeros((self.n_population, 2))
            delta = np.random.normal(size=(self.n_population,) + self.M.shape)
            done = np.full((self.n_population * 2,), False)
            obs = env.reset()
            while not done.all():
                actions = []
                for k in range(self.n_population):
                    for direction in range(2):
                        if not done[k * 2 + direction]:
                            current_obs = obs[k * 2 + direction].reshape(-1)
                            if direction == 0:
                                action = self.getAction([current_obs], delta=(self.exploration_noise * delta[k]))[0]
                            else:
                                action = self.getAction([current_obs], delta=(-self.exploration_noise * delta[k]))[0]

                            actions.append(action)
                        else:
                            actions.append(None)  # do nothing, as we are done

                obs, reward, new_done, info = env.step(actions)
                step += self.n_population

                done = np.bitwise_or(done, new_done)

                # cumulate the reward for every enviroment that is not finished
                update_idx = ~(done.reshape(self.n_population, 2))
                r[update_idx] += (reward.reshape(self.n_population, 2))[update_idx]

                if callback is not None:
                    callback(locals(), globals())
                if (step / self.n_population + 1) % 500 == 0:
                    print("{} steps - {:.2f} FPS".format(step, step / (time.time() - start_time)))

            idx = np.argsort(np.max(r, axis=1))[::-1]

            delta_sum = 0
            for i in range(self.top_population):
                delta_sum += (r[idx[i], 0] - r[idx[i], 1]) * delta[idx[i]]
            # here, we need to be careful with the normalization of step_size, as the variance can be 0 on sparse reward
            self.M += (self.step_size /
                       max(self.top_population * np.std(r[idx[:self.top_population]]), 1 / self.max_step_amplitude) *
                       delta_sum)
