import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from baselines.common.running_mean_std import RunningMeanStd
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from environments.utils import makeEnv
from rl_baselines.visualize import loadCsv
from srl_priors.utils import printYellow


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


def computeMeanReward(log_dir, last_n_episodes):
    """
    Compute the mean reward for the last n episodes
    :param log_dir: (str)
    :param last_n_episodes: (int)
    :return: (bool, numpy array)
    """
    result, _ = loadCsv(log_dir)
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

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
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

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
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


def createEnvs(args, pytorch=False):
    """
    :param args: (argparse.Namespace Object)
    :param pytorch: (bool)
    :return: (Gym VecEnv)
    """
    envs = [makeEnv(args.env, args.seed, i, args.log_dir)
            for i in range(args.num_cpu)]

    if len(envs) == 1:
        # No need for subprocesses when having only one env
        envs = DummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    envs = VecFrameStack(envs, args.num_stack)

    if args.srl_model != "":
        printYellow("Using MLP policy because working on state representation")
        args.policy = "mlp"
        envs = CustomVecNormalize(envs, norm_obs=True, norm_rewards=False)
    return envs
