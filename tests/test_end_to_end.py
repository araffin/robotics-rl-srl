from __future__ import print_function, division, absolute_import

import subprocess
import os
import json
from collections import OrderedDict

from srl_zoo.utils import createFolder

DEFAULT_ALGO = "ppo2"
DEFAULT_ENV = "KukaButtonGymEnv-v0"
DEFAULT_SRL = "supervised"
NUM_ITERATION = 1
NUM_TIMESTEP = 1600
DEFAULT_SRL_CONFIG_YAML = "config/srl_models_test.yaml"

DATA_FOLDER_NAME = "RL_test"
TEST_DATA_FOLDER = "data/" + DATA_FOLDER_NAME
LOG_FOLDER = "logs/RL_test/test_priors_custom_cnn/"
NUM_EPOCHS = 1
STATE_DIM = 3
TRAINING_SET_SIZE = 2000
KNN_SAMPLES = 1000

SEED = 0


def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)


def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)


def createFolders():
    createFolder("srl_zoo/" + LOG_FOLDER, "Test log folder already exist")
    folder_path = 'srl_zoo/{}/NearestNeighbors/'.format(LOG_FOLDER)
    createFolder(folder_path, "NearestNeighbors folder already exist")


def testDataGen():
    args = ['--num-cpu', 4, '--num-episode', 8, '--name', DATA_FOLDER_NAME, '--force', '--env', DEFAULT_ENV]
    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'environments.dataset_generator'] + args)
    assertEq(ok, 0)


def testBaselineTrain():
    for baseline in ['vae', 'autoencoder', 'supervised']:
        args = ['--no-plots', '--data-folder', TEST_DATA_FOLDER,
                '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                '--seed', SEED, '--model-type', 'cnn']
        if baseline != 'supervised':
            args += ['--state-dim', STATE_DIM]
        args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'baselines.{}'.format(baseline)] + args,  cwd=os.getcwd() + "/srl_zoo")
        assertEq(ok, 0)


def testPriorTrain():
    createFolders()
    for model_type in ['custom_cnn']:
        args = ['--no-plots', '--data-folder', TEST_DATA_FOLDER,
                '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                '--seed', SEED, '--val-size', 0.1, '--log-folder', LOG_FOLDER,
                '--state-dim', STATE_DIM, '--model-type', model_type, '-bs', 128]
        args = list(map(str, args))

        ok = subprocess.call(['python', 'train.py'] + args,  cwd=os.getcwd() + "/srl_zoo")
        assertEq(ok, 0)

        exp_config = {
            "batch-size": 128,
            "epochs": NUM_EPOCHS,
            "knn-samples": KNN_SAMPLES,
            "knn-seed": 1,
            "l1-reg": 0,
            "training-set-size": TRAINING_SET_SIZE,
            "learning-rate": 0.001,
            "model-approach": "priors",
            "model-type": model_type,
            "n-neighbors": 5,
            "n-to-plot": 5,
            "priors": [
                "Proportionality",
                "Temporal",
                "Causality",
                "Repeatability"
            ],
            "data-folder": TEST_DATA_FOLDER,
            "relative-pos": False,
            "seed": SEED,
            "state-dim": STATE_DIM,
            "log-folder": LOG_FOLDER,
            "experiment-name": "test_priors_custom_cnn",
            "use-continuous": False
        }
        exp_config = OrderedDict(sorted(exp_config.items()))
        with open("{}/exp_config.json".format("srl_zoo/" + exp_config['log-folder']), "w") as f:
            json.dump(exp_config, f)


def testRLSrlTrain():
    for model_type in ['vae', 'autoencoder', 'supervised', 'srl_priors']:
        args = ['--algo', DEFAULT_ALGO, '--env', DEFAULT_ENV, '--srl-model', model_type,
                '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--num-iteration', NUM_ITERATION,
                '--no-vis', '--srl-config-file', DEFAULT_SRL_CONFIG_YAML]
        args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
        assertEq(ok, 0)

    for algo in ['acer', 'deepq', 'a2c', 'ppo2', 'random_agent', 'ddpg', 'cma-es', 'ars']:
        args = ['--algo', algo, '--env', DEFAULT_ENV, '--srl-model', DEFAULT_SRL,
                '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--num-iteration', NUM_ITERATION,
                '--no-vis', '--srl-config-file', DEFAULT_SRL_CONFIG_YAML]
        if algo == "ddpg":
            mem_limit = 100 if model_type == 'raw_pixels' else 100000
            args.extend(['-c', '--memory-limit', mem_limit])
        elif algo == "acer":
            args.extend(['--num-stack', 4])

        args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
        assertEq(ok, 0)
