from __future__ import print_function, division, absolute_import

import subprocess

import pytest

from environments import ThreadingType
from environments.registry import registered_env

DEFAULT_ALGO = "ppo2"
DEFAULT_ENV = "MobileRobotGymEnv-v0"
DEFAULT_SRL = "ground_truth"
NUM_ITERATION = 1
NUM_TIMESTEP = 1600  # this should be long enough to call a reset of the environment
SEED = 0


def isXAvailable():
    """
    check to see if running in terminal with X or not
    :return: (bool)
    """
    p = subprocess.Popen(["xset", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    return p.returncode == 0


def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)


def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)


# ignoring 'trpo', as it will run out of memory and crash tensorflow's allocation
@pytest.mark.parametrize("algo", ['a2c', 'acer', 'acktr', 'ars', 'cma-es', 'ddpg', 'deepq', 'ppo1', 'ppo2', 'random_agent', 'sac'])
@pytest.mark.parametrize("model_type", ['raw_pixels'])
def testBaselineTrain(algo, model_type):
    """
    test for the given RL algorithm
    :param algo: (str) RL algorithm name
    :param model_type: (str) the model type to test
    """
    args = ['--algo', algo, '--srl-model', model_type, '--num-timesteps', NUM_TIMESTEP, '--seed', SEED,
            '--num-iteration', NUM_ITERATION, '--no-vis', '--env', DEFAULT_ENV]
    if algo == "ddpg":
        mem_limit = 100 if model_type == 'raw_pixels' else 100000
        args.extend(['-c', '--memory-limit', mem_limit])
    elif algo == "acer":
        args.extend(['--num-stack', 4])

    if algo in ["acer", "a2c", "ppo2"]:
        args.extend(["--num-cpu", 4])

    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
    assertEq(ok, 0)


@pytest.mark.parametrize("model_type", ['ground_truth', 'raw_pixels', 'joints', 'joints_position'])
@pytest.mark.parametrize("env", ["KukaButtonGymEnv-v0", "MobileRobotGymEnv-v0", "CarRacingGymEnv-v0"])
def testEnvSRLTrain(model_type, env):
    """
    test the environment states model on RL algorithms
    :param model_type: (str) the model type to test
    :param env: (str) the environment type to test
    """
    if env in ["CarRacingGymEnv-v0"] and not isXAvailable():
        return

    if model_type in ['joints', 'joints_position'] and env != "KukaButtonGymEnv-v0":
        return

    args = ['--algo', DEFAULT_ALGO, '--env', env, '--srl-model', model_type, '--num-timesteps', NUM_TIMESTEP,
            '--seed', SEED, '--num-iteration', NUM_ITERATION, '--no-vis']
    if registered_env[env][3] != ThreadingType.NONE:
        args.extend(['--num-cpu', 4])
    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
    assertEq(ok, 0)


@pytest.mark.fast
@pytest.mark.parametrize("env", ["KukaRandButtonGymEnv-v0", "Kuka2ButtonGymEnv-v0", "KukaMovingButtonGymEnv-v0",
                                 "MobileRobot2TargetGymEnv-v0", "MobileRobot1DGymEnv-v0", "MobileRobotLineTargetGymEnv-v0"])
def testEnvTrain(env):
    """
    test the environment on the RL pipeline
    :param env: (str) the environment type to test
    """
    args = ['--algo', DEFAULT_ALGO, '--env', env, '--srl-model', DEFAULT_SRL, '--num-timesteps', NUM_TIMESTEP,
            '--seed', SEED, '--num-iteration', NUM_ITERATION, '--no-vis']
    if registered_env[env][3] != ThreadingType.NONE:
        args.extend(['--num-cpu', 4])
    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
    assertEq(ok, 0)


@pytest.mark.fast
@pytest.mark.parametrize("env", ["KukaButtonGymEnv-v0", "MobileRobotGymEnv-v0", "CarRacingGymEnv-v0"])
@pytest.mark.parametrize("algo", ['a2c', 'ppo1', 'ppo2', 'sac', 'trpo'])
def testContinousEnvTrain(env, algo):
    """
    test the environment on the RL pipeline with continuous actions
    :param env: (str) the environment type to test
    :param algo: (str) RL algorithm name
    """
    if env in ["CarRacingGymEnv-v0"] and not isXAvailable():
        return

    args = ['--algo', algo, '--env', env, '--srl-model', DEFAULT_SRL, '--num-timesteps', NUM_TIMESTEP,
            '--seed', SEED, '--num-iteration', NUM_ITERATION, '--no-vis', '-c']
    if algo in ['ppo2'] and registered_env[env][3] != ThreadingType.NONE:
        args.extend(['--num-cpu', 4])
    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
    assertEq(ok, 0)
