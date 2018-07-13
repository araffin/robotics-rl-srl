from __future__ import print_function, division, absolute_import

import subprocess

DEFAULT_ALGO = "ppo2"
DEFAULT_ENV = "MobileRobotGymEnv-v0"
DEFAULT_SRL = "ground_truth"
NUM_ITERATION = 1
NUM_TIMESTEP = 1600  # this should be long enough to call a reset of the environment
SEED = 0


def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)


def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)


def makeTestFun(algo):
    """
    returns a test function for the given RL algorithm
    :param algo: (str) RL algorithm name
    :return: (func) A test function
    """
    def testBaselineTrain():
        for model_type in ['raw_pixels']:
            args = ['--algo', algo, '--srl-model', model_type, '--num-timesteps', NUM_TIMESTEP, '--seed', SEED,
                    '--num-iteration', NUM_ITERATION, '--no-vis']
            if algo == "ddpg":
                mem_limit = 100 if model_type == 'raw_pixels' else 100000
                args.extend(['-c', '--memory-limit', mem_limit])
                args.extend(['--env', "KukaButtonGymEnv-v0"])
            elif algo == "acer":
                args.extend(['--num-stack', 4])
                args.extend(['--env', DEFAULT_ENV])
            else:
                args.extend(['--env', DEFAULT_ENV])

            if algo in ["acer", "a2c", "ppo2"]:
                args.extend(["--num-cpu", 4])

            args = list(map(str, args))

            ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
            assertEq(ok, 0)

    return testBaselineTrain


for baseline in ['acer', 'deepq', 'a2c', 'ppo2', 'random_agent', 'ddpg', 'cma-es', 'ars', 'sac']:
    if baseline != DEFAULT_ALGO:
        globals()["test{}Train".format(baseline)] = makeTestFun(baseline)


def testEnvSRLTrain():
    for model_type in ['ground_truth', 'raw_pixels', 'joints', 'joints_position']:
        for env in ["KukaButtonGymEnv-v0", "MobileRobotGymEnv-v0"]:
            if model_type in ['joints', 'joints_position'] and env == "MobileRobotGymEnv-v0":
                continue

            args = ['--algo', DEFAULT_ALGO, '--env', env, '--srl-model', model_type, '--num-timesteps', NUM_TIMESTEP,
                    '--seed', SEED, '--num-iteration', NUM_ITERATION, '--no-vis', '--num-cpu', 4]
            args = list(map(str, args))

            ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
            assertEq(ok, 0)


def testEnvTrain():
    for env in ["KukaRandButtonGymEnv-v0", "Kuka2ButtonGymEnv-v0", "KukaMovingButtonGymEnv-v0",
                "MobileRobot2TargetGymEnv-v0", "MobileRobot1DGymEnv-v0", "MobileRobotLineTargetGymEnv-v0"]:
        args = ['--algo', DEFAULT_ALGO, '--env', env, '--srl-model', DEFAULT_SRL, '--num-timesteps', NUM_TIMESTEP,
                '--seed', SEED, '--num-iteration', NUM_ITERATION, '--no-vis', '--num-cpu', 4]
        args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
        assertEq(ok, 0)


def testContinousEnvTrain():
    for env in ["KukaButtonGymEnv-v0", "MobileRobotGymEnv-v0"]:
        for algo in ['ppo2', 'sac']:
            args = ['--algo', algo, '--env', env, '--srl-model', DEFAULT_SRL, '--num-timesteps', NUM_TIMESTEP,
                    '--seed', SEED, '--num-iteration', NUM_ITERATION, '--no-vis', '-c']
            if algo in ['ppo2']:
                args.extend(['--num-cpu', 4,])
            args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
        assertEq(ok, 0)
