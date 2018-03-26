import argparse
import os
import json
from datetime import datetime

import yaml
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from pytorch_agents.envs import make_env
import environments.kuka_button_gym_env as kuka_env
from rl_baselines.deepq import CustomDummyVecEnv, WrapFrameStack
from srl_priors.utils import printGreen, printYellow


def parseArguments(supported_models, pytorch=False, log_dir="/tmp/gym/test/"):
    """
    :param supported_models: ([str])
    :param pytorch: (bool)
    :param log_dir: (str) Log dir for testing the agent
    :return: (Arguments, dict, str, str, str, SubprocVecEnv)
    """
    parser = argparse.ArgumentParser(description="Load trained RL model")
    parser.add_argument('--env', help='environment ID', default='KukaButtonGymEnv-v0')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
    parser.add_argument('--log-dir', help='folder with the saved agent model', required=True)
    parser.add_argument('--num-timesteps', type=int, default=int(10e3))
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the environment (show the GUI)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA (works only with pytorch models)')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    load_args = parser.parse_args()
    # load_args.cuda = not load_args.no_cuda and th.cuda.is_available()

    with open('config/srl_models.yaml', 'rb') as f:
        srl_models = yaml.load(f)

    for algo in supported_models + ['not_supported']:
        if algo in load_args.log_dir:
            break
    if algo == "not_supported":
        raise ValueError("RL algo not supported for replay")
    printGreen("\n" + algo + "\n")

    extension = 'pth' if pytorch else 'pkl'
    load_path = "{}/{}_model.{}".format(load_args.log_dir, algo, extension)

    env_globals = json.load(open(load_args.log_dir + "kuka_env_globals.json", 'r'))
    train_args = json.load(open(load_args.log_dir + "args.json", 'r'))

    kuka_env.FORCE_RENDER = load_args.render
    kuka_env.ACTION_REPEAT = env_globals['ACTION_REPEAT']
    # Reward sparse or shaped
    kuka_env.SHAPE_REWARD = load_args.shape_reward

    if train_args["srl_model"] != "":
        train_args["policy"] = "mlp"
        path = srl_models.get(train_args["srl_model"])

        if train_args["srl_model"] == "ground_truth":
            kuka_env.USE_GROUND_TRUTH = True
        elif path is not None:
            kuka_env.USE_SRL = True
            kuka_env.SRL_MODEL_PATH = srl_models['log_folder'] + path
        else:
            raise ValueError("Unsupported value for srl-model: {}".format(train_args["srl_model"]))

    # Log dir for testing the agent
    log_dir += "{}/{}/".format(algo, datetime.now().strftime("%y-%m-%d_%Hh%M_%S"))
    os.makedirs(log_dir, exist_ok=True)

    if pytorch:
        if kuka_env.USE_SRL and not load_args.no_cuda:
            assert load_args.num_cpu == 1, "Multiprocessing not supported for srl models with CUDA (for pytorch_agents)"

        envs = [make_env(train_args['env'], load_args.seed, i, log_dir, pytorch=True)
                for i in range(load_args.num_cpu)]
        if load_args.num_cpu == 1:
            envs = DummyVecEnv(envs)
        else:
            envs = SubprocVecEnv(envs)
    else:
        if algo != "deepq":
            envs = SubprocVecEnv([make_env(train_args['env'], load_args.seed, i, log_dir, pytorch=False)
                                  for i in range(load_args.num_cpu)])
            envs = VecFrameStack(envs, train_args['num_stack'])
        else:
            if load_args.num_cpu > 1:
                printYellow("Deepq does not support multiprocessing, setting num-cpu=1")
            envs = CustomDummyVecEnv([make_env(train_args['env'], load_args.seed, 0, log_dir, pytorch=False)])
            # Normalize only raw pixels
            normalize = train_args['srl_model'] == ""
            envs = WrapFrameStack(envs, train_args['num_stack'], normalize=normalize)

    return load_args, train_args, load_path, log_dir, algo, envs
