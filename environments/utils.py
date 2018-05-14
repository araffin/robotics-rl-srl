# Modified version of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/envs.py

import os

import gym
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from gym.envs.registration import registry, patch_deprecated_methods, load


def makeEnv(env_id, seed, rank, log_dir, allow_early_resets=False, env_kwargs=None):
    """
    Instantiate gym env
    :param env_id: (str)
    :param seed: (int)
    :param rank: (int)
    :param log_dir: (str)
    :param allow_early_resets: (bool) Allow reset before the enviroment is done
    :param env_kwargs: (dict) The extra arguments for the environment
    """

    # define a place holder function to be returned to the caller.
    def _thunk():
        env = _make(env_id, env_kwargs=env_kwargs)
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


def _make(id_, env_kwargs=None):
    """
    Recreating the gym make function from gym/envs/registration.py
    as such as it can support extra arguments for the environment
    :param id_: (str) The environment ID
    :param env_kwargs: (dict) The extra arguments for the environment
    """
    if env_kwargs is None:
        env_kwargs = {}

    # getting the spec from the ID we want
    spec = registry.spec(id_)

    # Keeping the checks and safe guards of the old code
    assert spec._entry_point is not None, 'Attempting to make deprecated env {}. '\
            '(HINT: is there a newer registered version of this env?)'.format(spec.id_)

    if callable(spec._entry_point):
        env = spec._entry_point(**env_kwargs)
    else:
        cls = load(spec._entry_point)
        # create the env, with the original kwargs, and the new ones overriding them if needed
        env = cls(**{**spec._kwargs, **env_kwargs})

    # Make the enviroment aware of which spec it came from.
    env.unwrapped.spec = spec

    # Keeping the old patching system for _reset, _step and timestep limit
    if hasattr(env, "_reset") and hasattr(env, "_step") and not getattr(env, "_gym_disable_underscore_compat", False):
        patch_deprecated_methods(env)
    if (env.spec.timestep_limit is not None) and not spec.tags.get('vnc'):
        from gym.wrappers.time_limit import TimeLimit
        env = TimeLimit(env,
                        max_episode_steps=env.spec.max_episode_steps,
                        max_episode_seconds=env.spec.max_episode_seconds)
    return env

