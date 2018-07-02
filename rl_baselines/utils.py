import pickle
from collections import OrderedDict
from multiprocessing import Queue, Process

import numpy as np
import tensorflow as tf
import torch as th
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.vec_env import VecEnvWrapper, VecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack as OpenAIVecFrameStack

from environments.utils import makeEnv, dynamicEnvLoad
from rl_baselines.visualize import loadCsv
from srl_zoo.utils import printYellow, printGreen
from state_representation.models import loadSRLModel, getSRLDim


def createTensorflowSession():
    """
    Create tensorflow session with specific argument
    to prevent it from taking all gpu memory
    """
    # Let Tensorflow choose the device
    config = tf.ConfigProto(allow_soft_placement=True)
    # Prevent tensorflow from taking all the gpu memory
    config.gpu_options.allow_growth = True
    tf.Session(config=config).__enter__()


def computeMeanReward(log_dir, last_n_episodes, is_es=False):
    """
    Compute the mean reward for the last n episodes
    :param log_dir: (str)
    :param last_n_episodes: (int)
    :param is_es: (bool)
    :return: (bool, numpy array)
    """
    result, _ = loadCsv(log_dir, is_es=is_es)
    if len(result) == 0:
        return False, 0
    y = np.array(result)[:, 1]
    return True, y[-last_n_episodes:].mean()


def isJsonSafe(data):
    """
    Check if an object is json serializable
    :param data: (python object)
    :return: (bool)
    """
    if data is None:
        return True
    elif isinstance(data, (bool, int, float, str)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(isJsonSafe(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and isJsonSafe(v) for k, v in data.items())
    return False


def filterJSONSerializableObjects(input_dict):
    """
    Filter and sort entries of a dictionnary
    to save it as a json
    :param input_dict: (dict)
    :return: (OrderedDict)
    """
    output_dict = OrderedDict()
    for key in sorted(input_dict.keys()):
        if isJsonSafe(input_dict[key]):
            output_dict[key] = input_dict[key]
    return output_dict


class CustomVecNormalize(VecEnvWrapper):
    """
    Custom vectorized environment, it adds support for saving/loading moving average
    It can normalize observation and reward by computing a moving average
    :param venv: (VecEnv Object)
    :param training: (bool) Whether to update or not the moving average
    :param norm_obs: (bool) Whether to normalize observation or not (default: True)
    :param norm_rewards: (bool) Whether to normalize rewards or not (default: False)
    :param clip_obs: (float) Max absolute value for observation
    :param clip_reward: (float) Max value absolute for discounted reward
    :param gamma: (float) discount factor
    :param epsilon: (float) To avoid division by zero
    """

    def __init__(self, venv, training=True, norm_obs=True, norm_rewards=False,
                 clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_rewards = norm_rewards
        self.old_obs = np.array([])

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        self.old_obs = obs
        obs = self._normalizeObservation(obs)
        if self.norm_rewards:
            if self.training:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return obs, rews, news, infos

    def _normalizeObservation(self, obs):
        """
        :param obs: (numpy tensor)
        """
        if self.norm_obs:
            if self.training:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs,
                          self.clip_obs)
            return obs
        else:
            return obs

    def getOriginalObs(self):
        """
        retruns the unnormalized observation
        :return: (numpy float) 
        """
        return self.old_obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        if len(np.array(obs).shape) == 1:  # for when num_cpu is 1
            self.old_obs = [obs]
        else:
            self.old_obs = obs
        return self._normalizeObservation(obs)

    def saveRunningAverage(self, path):
        """
        :param path: (str) path to log dir
        """
        for rms, name in zip([self.obs_rms, self.ret_rms], ['obs_rms', 'ret_rms']):
            with open("{}/{}.pkl".format(path, name), 'wb') as f:
                pickle.dump(rms, f)

    def loadRunningAverage(self, path):
        """
        :param path: (str) path to log dir
        """
        for name in ['obs_rms', 'ret_rms']:
            with open("{}/{}.pkl".format(path, name), 'rb') as f:
                setattr(self, name, pickle.load(f))


class VecFrameStack(OpenAIVecFrameStack):
    """
    Vectorized environment class, fixed from OpenAIVecFrameStack
    :param venv: (Gym env)
    :param nstack: (int)
    """

    def __init__(self, venv, nstack):
        super(VecFrameStack, self).__init__(venv, nstack)

    def step_wait(self):
        """
        Step for each env
        :return: ([float], [float], [bool], dict) obs, reward, done, info
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[-1], axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos


class CustomDummyVecEnv(VecEnv):
    """Dummy class in order to use FrameStack with DQN"""

    def __init__(self, env_fns):
        """
        :param env_fns: ([function])
        """
        assert len(env_fns) == 1, "This dummy class does not support multiprocessing"
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.env = self.envs[0]
        self.actions = None
        self.obs = None
        self.reward, self.done, self.infos = None, None, None

    def step_wait(self):
        self.obs, self.reward, self.done, self.infos = self.env.step(self.actions[0])
        return self.obs[None], self.reward, [self.done], [self.infos]

    def step_async(self, actions):
        """
        :param actions: ([int])
        """
        self.actions = actions

    def reset(self):
        return self.env.reset()

    def close(self):
        return


class WrapFrameStack(VecFrameStack):
    """
    Wrap VecFrameStack in order to be usable with dqn
    and scale output if necessary
    """

    def __init__(self, venv, nstack, normalize=True):
        super(WrapFrameStack, self).__init__(venv, nstack)
        self.factor = 255.0 if normalize else 1

    def step(self, action):
        self.step_async([action])
        stackedobs, rews, news, infos = self.step_wait()
        return stackedobs[0] / self.factor, rews, news[0], infos[0]

    def reset(self):
        """
        Reset all environments
        """
        stackedobs = super(WrapFrameStack, self).reset()
        return stackedobs[0] / self.factor

    def getOriginalObs(self):
        """
        Hack to use CustomVecNormalize
        :return: (numpy float)
        """
        return self.venv.getOriginalObs()

    def saveRunningAverage(self, path):
        """
        Hack to use CustomVecNormalize
        :param path: (str) path to log dir
        """
        self.venv.saveRunningAverage(path)

    def loadRunningAverage(self, path):
        """
        Hack to use CustomVecNormalize
        :param path: (str) path to log dir
        """
        self.venv.loadRunningAverage(path)


class MultiprocessSRLModel:
    """
    Allows multiple environments to use a single SRL model
    :param num_cpu: (int) the number of environments that will spawn
    :param env_id: (str) the environment id string
    :param env_kwargs: (dict)
    """

    def __init__(self, num_cpu, env_id, env_kwargs):
        # Create a duplex pipe between env and srl model, where all the inputs are unified and the origin
        # marked with a index number
        self.pipe = (Queue(), [Queue() for _ in range(num_cpu)])
        module_env, class_name, _ = dynamicEnvLoad(env_id)
        # we need to know the expected dim output of the SRL model, before it is created
        self.state_dim = getSRLDim(env_kwargs.get("srl_model_path", None), module_env.__dict__[class_name])
        self.p = Process(target=self._run, args=(env_kwargs,))
        self.p.daemon = True
        self.p.start()

    def _run(self, env_kwargs):
        # this is to control the number of CPUs that torch is allowed to use.
        # By default it will use all CPUs, even with GPU acceleration
        th.set_num_threads(1)
        self.model = loadSRLModel(env_kwargs.get("srl_model_path", None), th.cuda.is_available(), self.state_dim, None)
        # run until the end of the caller thread
        while True:
            # pop an item, get state, and return to sender.
            env_id, var = self.pipe[0].get()
            self.pipe[1][env_id].put(self.model.getState(var))


def createEnvs(args, allow_early_resets=False, env_kwargs=None, load_path_normalise=None):
    """
    :param args: (argparse.Namespace Object)
    :param allow_early_resets: (bool) Allow reset before the enviroment is done, usually used in ES to halt the envs
    :param env_kwargs: (dict) The extra arguments for the environment
    :param load_path_normalise: (str) the path to loading the rolling average, None if not available or wanted.
    :return: (Gym VecEnv)
    """
    if env_kwargs is not None and env_kwargs.get("use_srl", False):
        srl_model = MultiprocessSRLModel(args.num_cpu, args.env, env_kwargs)
        env_kwargs["state_dim"] = srl_model.state_dim
        env_kwargs["srl_pipe"] = srl_model.pipe
    envs = [makeEnv(args.env, args.seed, i, args.log_dir, allow_early_resets=allow_early_resets, env_kwargs=env_kwargs)
            for i in range(args.num_cpu)]

    if len(envs) == 1:
        # No need for subprocesses when having only one env
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    envs = VecFrameStack(envs, args.num_stack)

    if args.srl_model != "raw_pixels":
        printYellow("Using MLP policy because working on state representation")
        # set the value MLP for any RL model that dont define a policy (or askes for CNN policy)
        # and are not using raw_pixels, or asked for a linear policy.
        if not hasattr(args, "policy") or args.policy == "linear" or args.policy == "cnn":
            args.policy = "mlp"
        envs = CustomVecNormalize(envs, norm_obs=True, norm_rewards=False)
        envs = loadRunningAverage(envs, load_path_normalise=load_path_normalise)
    else:
        # set the value CNN for any RL model that dont define it
        # and are using raw_pixels, or asked for a linear policy.
        if not hasattr(args, "policy") or args.policy == "linear":
            args.policy = "cnn"
        elif "cnn" not in args.policy:  # extend for those who have something else with CNN (eg: cnn-lstm)
            args.policy = "cnn-" + args.policy

    return envs


def loadRunningAverage(envs, load_path_normalise=None):
    if load_path_normalise is not None:
        try:
            printGreen("Loading saved running average")
            envs.loadRunningAverage(load_path_normalise)
            envs.training = False
        except FileNotFoundError:
            envs.training = True
            printYellow("Running Average files not found for CustomVecNormalize, switching to training mode")
    return envs


def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x.T - np.max(x.T, axis=0))
    return (e_x / e_x.sum(axis=0)).T

