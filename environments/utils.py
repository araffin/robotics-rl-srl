# Modified version of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/envs.py

import os

import gym

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind


def makeEnv(env_id, seed, rank, log_dir, allow_early_resets=False):
    """
    Instantiate gym env
    :param env_id: (str)
    :param seed: (int)
    :param rank: (int)
    :param log_dir: (str)
    :param allow_early_resets: (bool) Allow reset before the enviroment is done
    """

    def _thunk():
        env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)
        if is_atari:
            env = wrap_deepmind(env)
        return env

    return _thunk

