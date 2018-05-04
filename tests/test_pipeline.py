from __future__ import print_function, division, absolute_import

import subprocess

DEFAULT_ALGO = "ppo2"
DEFAULT_ENV = "KukaButtonGymEnv-v0"
DEFAULT_SRL = "ground_truth"
NUM_ITERATION = 1
NUM_TIMESTEP = 1600
SEED = 0

def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)

def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)

# def testSrlTrain():
#     for model_type in ["ground_truth", "srl_priors", "vae"]:
#         args = ['--algo', DEFAULT_ALGO, '--env', DEFAULT_ENV, '--srl-model', model_type, 
#                 '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--num-iteration', NUM_ITERATION,
#                 '--no-vis']
#         args = list(map(str, args))

#         ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
#         assertEq(ok, 0)


def make_test_fun(algo):
    def testBaselineTrain():
        algo_baseline = algo
        for model_type in ['ground_truth', 'raw_pixels']:
            args = ['--algo', algo, '--env', DEFAULT_ENV, '--srl-model', model_type, 
                    '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--num-iteration', NUM_ITERATION,
                    '--no-vis']
            if algo == "ddpg":
                mem_limit = 100 if model_type == 'raw_pixels' else 100000
                args.extend(['-c', '--memory-limit', mem_limit])
            elif algo == "acer":
                args.extend(['--num-stack', 4])

            args = list(map(str, args))

            ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
            assertEq(ok, 0)
    return testBaselineTrain

for baseline in ['acer', 'deepq', 'a2c', 'ppo2', 'random_agent', 'ddpg', 'cma-es', 'ars']:
    globals()["test"+baseline+"Train"] = make_test_fun(baseline)


def testEnvTrain():
    for env in ["KukaButtonGymEnv-v0", "KukaRandButtonGymEnv-v0", "Kuka2ButtonGymEnv-v0", "KukaMovingButtonGymEnv-v0"]:
        args = ['--algo', DEFAULT_ALGO, '--env', env, '--srl-model', DEFAULT_SRL, 
                '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--num-iteration', NUM_ITERATION,
                '--no-vis']
        args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
        assertEq(ok, 0)