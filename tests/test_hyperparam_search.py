from __future__ import print_function, division, absolute_import

import subprocess

import pytest

DEFAULT_OPTIMIZER = 'hyperband'
DEFAULT_ALGO = "ppo2"
DEFAULT_ENV = "MobileRobotGymEnv-v0"
DEFAULT_SRL = "ground_truth"
NUM_ITERATION = 1
NUM_TIMESTEP = 10000  # this should be long enough to call a reset of the environment
MAX_EVAL = 2  # hyperopt evals
SEED = 0


def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)


def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)


@pytest.mark.slow
@pytest.mark.parametrize("optimizer", ['hyperband', 'hyperopt'])
def testHyperparamOptimizer(optimizer):
    """
    test for the given hyperparam optimizer
    :param optimizer: (str) RL algorithm name
    """
    args = ['--optimizer', optimizer, '--algo', DEFAULT_ALGO, '--srl-model', DEFAULT_SRL, '--max-eval', MAX_EVAL,
            '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--env', DEFAULT_ENV, "--num-cpu", 4]

    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.hyperparam_search'] + args)
    assertEq(ok, 0)


@pytest.mark.slow
@pytest.mark.parametrize("algo", ['acer', 'deepq', 'a2c', 'ddpg', 'cma-es', 'ars', 'sac'])
def testRLHyperparamSearch(algo):
    """
    test for the given RL algorithm
    :param algo: (str) RL algorithm name
    """
    args = ['--optimizer', DEFAULT_OPTIMIZER, '--algo', algo, '--srl-model', DEFAULT_SRL, '--max-eval', MAX_EVAL,
            '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--env', DEFAULT_ENV]
    if algo == "ddpg":
        args.extend(['-c', '--memory-limit', 100000])
    elif algo == "acer":
        args.extend(['--num-stack', 4])

    if algo in ["acer", "a2c", "ppo2"]:
        args.extend(["--num-cpu", 4])

    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.hyperparam_search'] + args)
    assertEq(ok, 0)
