from __future__ import print_function, division, absolute_import

import subprocess
import glob
import shutil

DEFAULT_ALGO = "ppo2"
DEFAULT_ENV = "KukaButtonGymEnv-v0"
DEFAULT_SRL = "ground_truth"
DEFAULT_LOG_DIR = "logs/test_enjoy/"
NUM_ITERATION = 1
NUM_TRAIN_TIMESTEP = 2000
NUM_ENJOY_TIMESTEP = 700
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


def testBaselineTrain():
    # cleanup to remove the cluter
    shutil.rmtree(DEFAULT_LOG_DIR)
    for algo in ['acer', 'deepq', 'a2c', 'ppo2', 'random_agent', 'ddpg', 'cma-es', 'ars']:
        args = ['--algo', algo, '--srl-model', DEFAULT_SRL, '--num-timesteps', NUM_TRAIN_TIMESTEP, '--seed', SEED,
                '--num-iteration', NUM_ITERATION, '--no-vis', '--log-dir', DEFAULT_LOG_DIR, '--env', DEFAULT_ENV]
        if algo == "ddpg":
            args.extend(['-c', '--memory-limit', 100])
        elif algo == "acer":
            args.extend(['--num-stack', 4])

        if algo in ["acer", "a2c", "ppo2"]:
            args.extend(['--num-cpu', 4])

        args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
        assertEq(ok, 0)

    return testBaselineTrain


def testEnjoyBaselines():
    if isXAvailable():
        for algo in ['acer', 'deepq', 'a2c', 'ppo2', 'ddpg', 'cma-es', 'ars']:
            directory = sorted(glob.glob("logs/test_enjoy/{}/{}/{}/*".format(DEFAULT_ENV, DEFAULT_SRL, algo)))[-1] + "/"

            args = ['--log-dir', directory, '--num-timesteps', NUM_ENJOY_TIMESTEP, '--plotting', '--action-proba']
            args = list(map(str, args))

            ok = subprocess.call(['python', '-m', 'replay.enjoy_baselines'] + args)
            assertEq(ok, 0)
    else:
        print("X not available, ignoring test")
