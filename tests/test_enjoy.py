from __future__ import print_function, division, absolute_import

import subprocess
import glob
import shutil
import os

import pytest

DEFAULT_ALGO = "ppo2"
DEFAULT_ENV = "KukaButtonGymEnv-v0"
DEFAULT_SRL = "ground_truth"
DEFAULT_LOG_DIR = "logs/test_enjoy/"
NUM_ITERATION = 1
NUM_TRAIN_TIMESTEP = 3000
NUM_ENJOY_TIMESTEP = 700
SEED = 0

# cleanup to remove the cluter
if os.path.exists(DEFAULT_LOG_DIR):
    shutil.rmtree(DEFAULT_LOG_DIR)


def isXAvailable():
    """
    check to see if running in terminal with X or not
    :return: (bool)
    """
    try:
        p = subprocess.Popen(["xset", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()
        return p.returncode == 0
    except FileNotFoundError:
        # Return False if xset is not present on the machine
        return False


def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)


def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)


@pytest.mark.fast
@pytest.mark.parametrize("algo", ['acer', 'deepq', 'a2c', 'ppo2', 'ddpg', 'cma-es', 'ars', 'sac'])
def testBaselineTrain(algo):
    """
    test for the given RL algorithm
    :param algo: (str) RL algorithm name
    """
    args = ['--algo', algo, '--srl-model', DEFAULT_SRL, '--num-timesteps', NUM_TRAIN_TIMESTEP, '--seed', SEED,
            '--num-iteration', NUM_ITERATION, '--no-vis', '--log-dir', DEFAULT_LOG_DIR, '--env', DEFAULT_ENV,
            '--min-episodes-save', 1]
    if algo == "ddpg":
        args.extend(['-c', '--memory-limit', 100])
    elif algo == "acer":
        args.extend(['--num-stack', 4])

    if algo in ["acer", "a2c", "ppo2"]:
        args.extend(['--num-cpu', 4])

    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
    assertEq(ok, 0)


@pytest.mark.slow
@pytest.mark.parametrize("algo", ['acer', 'deepq', 'a2c', 'ppo2', 'ddpg', 'cma-es', 'ars', 'sac'])
def testEnjoyBaselines(algo):
    """
    test the enjoy script for the given RL algorithm
    :param algo: (str) RL algorithm name
    """
    if isXAvailable():
        directory = sorted(glob.glob("logs/test_enjoy/{}/{}/{}/*".format(DEFAULT_ENV, DEFAULT_SRL, algo)))[-1] + "/"

        args = ['--log-dir', directory, '--num-timesteps', NUM_ENJOY_TIMESTEP, '--plotting', '--action-proba']
        args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'replay.enjoy_baselines'] + args)
        assertEq(ok, 0)
    else:
        print("X not available, ignoring test")
