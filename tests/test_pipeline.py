from __future__ import print_function, division, absolute_import

import subprocess

DEFAULT_ALGO = "ppo2"
DEFAULT_ENV = "KukaButtonGymEnv-v0"
DEFAULT_SRL = "ground_truth"
NUM_ITERATION = 1
NUM_TIMESTEP = 100
SEED = 0

def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)

def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)

# def testSrlTrain():
#     for model_type in ["ground_truth", "srl_priors", "vae"]:
#         args = ['--algo', DEFAULT_ALGO, '--env', DEFAULT_ENV, '--srl-model', model_type, 
#                 '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--num-iteration', NUM_ITERATION]
#         args = list(map(str, args))

#         ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
#         assertEq(ok, 0)

def testBaselineTrain():
    for model_type in ['ground_truth', 'raw_pixels']:
        for baseline in ['acer', 'deepq', 'a2c', 'ppo2', 'random_agent', 'ddpg', 'cma-es', 'ars']:
            args = ['--algo', baseline, '--env', DEFAULT_ENV, '--srl-model', model_type, 
                    '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--num-iteration', NUM_ITERATION]
            args = list(map(str, args))

            ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
            assertEq(ok, 0)

def testEnvTrain():
    for env in ["KukaButtonGymEnv-v0", "KukaRandButtonGymEnv-v0", "Kuka2ButtonGymEnv-v0", "KukaMovingButtonGymEnv-v0"]:
        args = ['--algo', DEFAULT_ALGO, '--env', env, '--srl-model', DEFAULT_SRL, 
                '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--num-iteration', NUM_ITERATION]
        args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
        assertEq(ok, 0)