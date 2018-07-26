from __future__ import print_function, division, absolute_import

import subprocess
import os
import json
from collections import OrderedDict

import pytest

from srl_zoo.utils import createFolder

DEFAULT_ALGO = "ppo2"
DEFAULT_ENV = "KukaButtonGymEnv-v0"
DEFAULT_SRL = "supervised"
NUM_ITERATION = 1
NUM_TIMESTEP = 1600
DEFAULT_SRL_CONFIG_YAML = "config/srl_models_test.yaml"

DATA_FOLDER_NAME = "RL_test"
TEST_DATA_FOLDER = "data/" + DATA_FOLDER_NAME
TEST_DATA_FOLDER_DUAL_CAMERA = "data/kuka_gym_dual_test"

NUM_EPOCHS = 1
STATE_DIM = 3
TRAINING_SET_SIZE = 2000
KNN_SAMPLES = 1000

SEED = 0


def buildTestConfig():
    cfg = {
        "batch-size": 32,
        "model-type": "custom_cnn",
        "epochs": NUM_EPOCHS,
        "knn-samples": KNN_SAMPLES,
        "knn-seed": 1,
        "l1-reg": 0,
        "training-set-size": TRAINING_SET_SIZE,
        "learning-rate": 0.001,
        "data-folder": TEST_DATA_FOLDER,
        "relative-pos": False,
        "seed": SEED,
        "state-dim": STATE_DIM,
        "use-continuous": False
    }
    return cfg


def assertEq(left, right):
    assert left == right, "{} != {}".format(left, right)


def assertNeq(left, right):
    assert left != right, "{} == {}".format(left, right)


def createFolders(log_folder_name):
    createFolder("srl_zoo/" + log_folder_name, "Test log folder already exist")
    folder_path = 'srl_zoo/{}/NearestNeighbors/'.format(log_folder_name)
    createFolder(folder_path, "NearestNeighbors folder already exist")


def testDataGen():
    args = ['--num-cpu', 4, '--num-episode', 8, '--name', DATA_FOLDER_NAME, '--force', '--env', DEFAULT_ENV,
            '--reward-dist']
    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'environments.dataset_generator'] + args)
    assertEq(ok, 0)


@pytest.mark.parametrize("baseline", ["supervised", "vae", "autoencoder"])
def testBaselineTrain(baseline):
    """
    Testing baseline models
    :param baseline: (str) the baseline name to test
    """
    if baseline == 'supervised':
        args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER,
                '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                '--seed', SEED, '--model-type', 'cnn']
        args = list(map(str, args))

        ok = subprocess.call(['python', '-m', 'srl_baselines.supervised'] + args, cwd=os.getcwd() + "/srl_zoo")
        assertEq(ok, 0)
    else:
        exp_name = baseline + '_cnn_ST_DIM3_SEED0_NOISE0_EPOCHS1_BS32'
        LOG_BASELINE = 'logs/' + DATA_FOLDER_NAME + '/' + exp_name
        createFolders(LOG_BASELINE)
        exp_config = buildTestConfig()
        exp_config["log-folder"] = LOG_BASELINE
        exp_config["experiment-name"] = exp_name
        exp_config["losses"] = baseline
        exp_config["batch-size"] = 32
        exp_config["n_actions"] = 6
        print("log baseline: ", LOG_BASELINE)
        args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER,
                '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
                '--seed', SEED, '--model-type', 'custom_cnn',
                '--state-dim', STATE_DIM, '-bs', 32,
                '--losses', baseline,
                '--log-folder', LOG_BASELINE]
        args = list(map(str, args))

        with open("{}/exp_config.json".format("srl_zoo/" + exp_config['log-folder']), "w") as f:
            json.dump(exp_config, f)
        ok = subprocess.call(['python', 'train.py'] + args, cwd=os.getcwd() + "/srl_zoo")
        assertEq(ok, 0)


@pytest.mark.parametrize("loss_type", ["priors", "inverse", "forward", "triplet"])
def testSrlTrain(loss_type):
    """
    Testing the training of srl models to be later used for RL
    :param loss_type: (str) the model loss to test
    """
    exp_name = loss_type + '_cnn_ST_DIM3_SEED0_NOISE0_EPOCHS1_BS32'
    log_name = 'logs/' + DATA_FOLDER_NAME + '/' + exp_name
    createFolders(log_name)
    exp_config = buildTestConfig()

    args = ['--no-display-plots', '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
            '--seed', SEED, '--val-size', 0.1, '--state-dim', STATE_DIM, '--model-type', 'custom_cnn', '-bs', 32,
            '--log-folder', log_name,'--losses', loss_type]

    # Testing multi-view
    if loss_type == "triplet":
        exp_config["multi-view"] = True
        args.extend(['--multi-view', '--data-folder', TEST_DATA_FOLDER_DUAL_CAMERA])
    else:
        args.extend(['--data-folder', TEST_DATA_FOLDER])

    args = list(map(str, args))

    exp_config["log-folder"] = log_name
    exp_config["experiment-name"] = exp_name
    exp_config["losses"] = loss_type
    exp_config["n_actions"] = 6
    exp_config = OrderedDict(sorted(exp_config.items()))
    with open("{}/exp_config.json".format("srl_zoo/" + exp_config['log-folder']), "w") as f:
        json.dump(exp_config, f)
    ok = subprocess.call(['python', 'train.py'] + args, cwd=os.getcwd() + "/srl_zoo")
    assertEq(ok, 0)


def testSrlCombiningTrain():
    # Combining models
    exp_name = 'vae_inverse_forward_cnn_ST_DIM3_SEED0_NOISE0_EPOCHS1_BS32'
    log_name = 'logs/' + DATA_FOLDER_NAME + '/' + exp_name
    createFolders(log_name)
    args = ['--no-display-plots', '--data-folder', TEST_DATA_FOLDER,
            '--epochs', NUM_EPOCHS, '--training-set-size', TRAINING_SET_SIZE,
            '--seed', SEED, '--val-size', 0.1, '--state-dim', STATE_DIM, '--model-type', 'custom_cnn', '-bs', 32,
            '--log-folder', log_name, '--losses', "forward", "inverse", "vae"]
    args = list(map(str, args))

    exp_config = buildTestConfig()
    exp_config["log-folder"] = log_name
    exp_config["experiment-name"] = exp_name
    exp_config["losses"] = ["forward", "inverse", "vae"]
    exp_config["n_actions"] = 6
    exp_config = OrderedDict(sorted(exp_config.items()))
    with open("{}/exp_config.json".format("srl_zoo/" + exp_config['log-folder']), "w") as f:
        json.dump(exp_config, f)
    ok = subprocess.call(['python', 'train.py'] + args, cwd=os.getcwd() + "/srl_zoo")
    assertEq(ok, 0)


@pytest.mark.parametrize("model_type", ['vae', 'autoencoder', "robotic_priors", "inverse", "forward", "srl_combination", "multi_view_srl"])
def testAllRLOnSrlTrain(model_type):
    """
    Testing all the previously learned srl models on the RL pipeline
    :param model_type: (str) the srl model to run
    """
    args = ['--algo', DEFAULT_ALGO, '--env', DEFAULT_ENV, '--srl-model', model_type,
            '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--num-iteration', NUM_ITERATION,
            '--no-vis', '--srl-config-file', DEFAULT_SRL_CONFIG_YAML]
    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
    assertEq(ok, 0)


@pytest.mark.parametrize("algo", ['acer', 'deepq', 'a2c', 'ppo2', 'random_agent', 'ddpg', 'cma-es', 'ars', 'sac'])
def testAllSrlonRLTrain(algo):
    """
    Testing RL pipeline on previously learned models
    :param algo: (str) RL algorithm name
    """
    args = ['--algo', algo, '--env', DEFAULT_ENV, '--srl-model', DEFAULT_SRL,
            '--num-timesteps', NUM_TIMESTEP, '--seed', SEED, '--num-iteration', NUM_ITERATION,
            '--no-vis', '--srl-config-file', DEFAULT_SRL_CONFIG_YAML]
    if algo == "ddpg":
        mem_limit = 100 if DEFAULT_SRL == 'raw_pixels' else 100000
        args.extend(['-c', '--memory-limit', mem_limit])
    elif algo == "acer":
        args.extend(['--num-stack', 4])

    args = list(map(str, args))

    ok = subprocess.call(['python', '-m', 'rl_baselines.pipeline'] + args)
    assertEq(ok, 0)
