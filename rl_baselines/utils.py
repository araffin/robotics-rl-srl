from collections import OrderedDict
from multiprocessing import Queue, Process

import numpy as np
import tensorflow as tf
import torch as th
from stable_baselines.common.vec_env import VecEnv, VecNormalize, DummyVecEnv, SubprocVecEnv, VecFrameStack

from environments import ThreadingType
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


def computeMeanReward(log_dir, last_n_episodes, is_es=False, return_n_episodes=False):
    """
    Compute the mean reward for the last n episodes
    :param log_dir: (str)
    :param last_n_episodes: (int)
    :param is_es: (bool)
    :param return_n_episodes: (bool)
    :return: (bool, numpy array or tuple when return_n_episodes is True)
    """
    result, _ = loadCsv(log_dir, is_es=is_es)
    if len(result) == 0:
        return False, 0
    y = np.array(result)[:, 1]

    if return_n_episodes:
        return True, (y[-last_n_episodes:].mean(), len(y))
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


class CustomDummyVecEnv(VecEnv):
    """Dummy class in order to use FrameStack with SAC"""

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
        return np.copy(self.obs[None]), self.reward, [self.done], [self.infos]

    def step_async(self, actions):
        """
        :param actions: ([int])
        """
        self.actions = actions

    def reset(self):
        return self.env.reset()

    def close(self):
        return

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]


class WrapFrameStack(VecFrameStack):
    """
    Wrap VecFrameStack in order to be usable with SAC
    and scale output if necessary
    """

    def __init__(self, venv, n_stack, normalize=True):
        super(WrapFrameStack, self).__init__(venv, n_stack)
        self.factor = 255.0 if normalize else 1

    def step(self, action):
        self.step_async([action])
        stackedobs, rewards, dones, infos = self.step_wait()
        return stackedobs[0] / self.factor, rewards, dones[0], infos[0]

    def reset(self):
        """
        Reset all environments
        """
        stackedobs = super(WrapFrameStack, self).reset()
        return stackedobs[0] / self.factor

    def get_original_obs(self):
        """
        Hack to use CustomVecNormalize
        :return: (numpy float)
        """
        return self.venv.get_original_obs()

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
        self.model = loadSRLModel(env_kwargs.get("srl_model_path", None), th.cuda.is_available(), self.state_dim,
                                  env_object=None)
        # run until the end of the caller thread
        while True:
            # pop an item, get state, and return to sender.
            env_id, var = self.pipe[0].get()
            self.pipe[1][env_id].put(self.model.getState(var, env_id=env_id))


def createEnvs(args, allow_early_resets=False, env_kwargs=None, load_path_normalise=None):
    """
    :param args: (argparse.Namespace Object)
    :param allow_early_resets: (bool) Allow reset before the enviroment is done, usually used in ES to halt the envs
    :param env_kwargs: (dict) The extra arguments for the environment
    :param load_path_normalise: (str) the path to loading the rolling average, None if not available or wanted.
    :return: (Gym VecEnv)
    """
    # imported here to prevent cyclic imports
    from environments.registry import registered_env
    from state_representation.registry import registered_srl, SRLType

    assert not (registered_env[args.env][3] is ThreadingType.NONE and args.num_cpu != 1), \
        "Error: cannot have more than 1 CPU for the environment {}".format(args.env)

    if env_kwargs is not None and registered_srl[args.srl_model][0] == SRLType.SRL:
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
        envs = VecNormalize(envs, norm_obs=True, norm_reward=False)
        envs = loadRunningAverage(envs, load_path_normalise=load_path_normalise)

    return envs


def loadRunningAverage(envs, load_path_normalise=None):
    if load_path_normalise is not None:
        try:
            printGreen("Loading saved running average")
            envs.load_running_average(load_path_normalise)
            envs.training = False
        except FileNotFoundError:
            envs.training = True
            printYellow("Running Average files not found for CustomVecNormalize, switching to training mode")
    return envs


def softmax(x):
    """
    Numerically stable implementation of softmax.
    :param x: (numpy float)
    :return: (numpy float)
    """
    e_x = np.exp(x.T - np.max(x.T, axis=0))
    return (e_x / e_x.sum(axis=0)).T
