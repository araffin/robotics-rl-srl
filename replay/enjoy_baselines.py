"""
Enjoy script for OpenAI Baselines
"""
import argparse
import json
import os
from datetime import datetime

import yaml
from baselines.acer.acer_simple import *
from baselines.acer.policies import AcerCnnPolicy
from baselines.ppo2.policies import CnnPolicy, MlpPolicy
from baselines.common import tf_util, set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines import deepq

import rl_baselines.ddpg as ddpg
import rl_baselines.ars as ars
import rl_baselines.cma_es as cma_es
from rl_baselines.deepq import CustomDummyVecEnv, WrapFrameStack
from rl_baselines.utils import createTensorflowSession, computeMeanReward, CustomVecNormalize, VecFrameStack
from rl_baselines.policies import MlpPolicyDicrete, AcerMlpPolicy, CNNPolicyContinuous
from srl_priors.utils import printYellow, printGreen
from environments.utils import makeEnv
from gazebo.constants import USING_REAL_BAXTER
if USING_REAL_BAXTER:
    import environments.gym_baxter.baxter_env as kuka_env
else:
    import environments.kuka_button_gym_env as kuka_env

supported_models = ['acer', 'ppo2', 'a2c', 'deepq', 'ddpg', 'ars', 'cma-es']


def parseArguments():
    """

    :return: (Arguments)
    """
    parser = argparse.ArgumentParser(description="Load trained RL model")
    parser.add_argument('--env', help='environment ID', type=str, default='KukaButtonGymEnv-v0')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
    parser.add_argument('--log-dir', help='folder with the saved agent model', type=str, required=True)
    parser.add_argument('--num-timesteps', type=int, default=int(1e4))
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the environment (show the GUI)')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    return parser.parse_args()


def loadConfigAndSetup(load_args):
    """
    Get the training config and setup the parameters
    :param load_args: (Arguments)
    :return: (dict, str, str)
    """
    with open('config/srl_models.yaml', 'rb') as f:
        srl_models = yaml.load(f)

    for algo in supported_models + ['not_supported']:
        if algo in load_args.log_dir:
            break
    if algo == "not_supported":
        raise ValueError("RL algo not supported for replay")
    printGreen("\n" + algo + "\n")

    load_path = "{}/{}_model.pkl".format(load_args.log_dir, algo)

    env_globals = json.load(open(load_args.log_dir + "kuka_env_globals.json", 'r'))
    train_args = json.load(open(load_args.log_dir + "args.json", 'r'))

    kuka_env.FORCE_RENDER = load_args.render
    kuka_env.ACTION_REPEAT = env_globals['ACTION_REPEAT']
    # Reward sparse or shaped
    kuka_env.SHAPE_REWARD = load_args.shape_reward

    kuka_env.ACTION_JOINTS = train_args["action_joints"]
    kuka_env.IS_DISCRETE = not train_args["continuous_actions"]
    kuka_env.BUTTON_RANDOM = train_args.get('relative', False)
    # Remove up action
    kuka_env.FORCE_DOWN = env_globals.get('FORCE_DOWN', True)

    if train_args["srl_model"] != "":
        train_args["policy"] = "mlp"
        path = srl_models.get(train_args["srl_model"])

        if train_args["srl_model"] == "ground_truth":
            kuka_env.USE_GROUND_TRUTH = True
        elif train_args["srl_model"] == "joints":
            kuka_env.USE_JOINTS = True
        elif train_args["srl_model"] == "joints_position":
            kuka_env.USE_GROUND_TRUTH = True
            kuka_env.USE_JOINTS = True
        elif path is not None:
            kuka_env.USE_SRL = True
            kuka_env.SRL_MODEL_PATH = srl_models['log_folder'] + path
        else:
            raise ValueError("Unsupported value for srl-model: {}".format(train_args["srl_model"]))

    return train_args, load_path, algo


def createEnv(load_args, train_args, algo, log_dir="/tmp/gym/test/"):
    """
    Create the Gym environment
    :param load_args: (Arguments)
    :param train_args: (dict)
    :param algo: (str)
    :param log_dir: (str) Log dir for testing the agent
    :return: (str, SubprocVecEnv)
    """
    # Log dir for testing the agent
    log_dir += "{}/{}/".format(algo, datetime.now().strftime("%y-%m-%d_%Hh%M_%S"))
    os.makedirs(log_dir, exist_ok=True)

    if algo not in ["deepq", "ddpg"]:
        envs = SubprocVecEnv([makeEnv(train_args['env'], load_args.seed, i, log_dir)
                              for i in range(load_args.num_cpu)])
        envs = VecFrameStack(envs, train_args['num_stack'])
    else:
        if load_args.num_cpu > 1:
            printYellow(algo + " does not support multiprocessing, setting num-cpu=1")
        envs = CustomDummyVecEnv([makeEnv(train_args['env'], load_args.seed, 0, log_dir)])

    if train_args["srl_model"] != "":
        # special case where ars type v1 is not normalized
        if algo != "ars" or train_args['algo_type'] != "v1":
            envs = CustomVecNormalize(envs, training=False)
        # Temp fix for experiments where no running average were saved
        try:
            printGreen("Loading saved running average")
            envs.loadRunningAverage(load_args.log_dir)
        except FileNotFoundError:
            envs.training = True
            printYellow("Running Average files not found for CustomVecNormalize, switching to training mode")

    if algo in ["deepq", "ddpg"]:
        # Normalize only raw pixels
        normalize = train_args['srl_model'] == ""
        envs = WrapFrameStack(envs, train_args['num_stack'], normalize=normalize)

    return log_dir, envs


def main():
    load_args = parseArguments()
    train_args, load_path, algo = loadConfigAndSetup(load_args)
    log_dir, envs = createEnv(load_args, train_args, algo)

    ob_space = envs.observation_space
    ac_space = envs.action_space

    tf.reset_default_graph()
    set_global_seeds(load_args.seed)
    createTensorflowSession()

    sess = tf_util.make_session()
    printYellow("Compiling Policy function....")
    if algo == "acer":
        policy = {'cnn': AcerCnnPolicy, 'mlp': AcerMlpPolicy}[train_args["policy"]]
        # nstack is already handled in the VecFrameStack
        model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, nstack=1, reuse=False)
    elif algo == "a2c":
        policy = {'cnn': CnnPolicy, 'mlp': MlpPolicyDicrete}[train_args["policy"]]
        model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, reuse=False)
    elif algo == "ppo2":
        if train_args["continuous_actions"]:
            policy = {'cnn': CNNPolicyContinuous, 'mlp': MlpPolicy}[train_args["policy"]]
        else:
            policy = {'cnn': CnnPolicy, 'mlp': MlpPolicyDicrete}[train_args["policy"]]
        model = policy(sess, ob_space, ac_space, load_args.num_cpu, nsteps=1, reuse=False)
    elif algo == "ddpg":
        model = ddpg.load(load_path, sess)
    elif algo == "ars":
        model = ars.load(load_path)
    elif algo == "cma-es":
        model = cma_es.load(load_path)

    params = find_trainable_variables("model")

    tf.global_variables_initializer().run(session=sess)

    # Load weights
    if algo in ["acer", "a2c", "ppo2"]:
        loaded_params = joblib.load(load_path)
        restores = []
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        ps = sess.run(restores)
    elif algo == "deepq":
        model = deepq.load(load_path)
    elif algo == "ddpg":
        model.load(load_path)
    elif algo == "cma-es":
        model.policy.setParam(model.best_model)

    dones = [False for _ in range(load_args.num_cpu)]
    obs = envs.reset()
    # print(obs.shape)

    n_done = 0
    last_n_done = 0
    for _ in range(load_args.num_timesteps):
        if algo == "acer":
            actions, state, _ = model.step(obs, state=None, mask=dones)
        elif algo in ["a2c", "ppo2"]:
            actions, _, states, _ = model.step(obs, None, dones)
        elif algo == "deepq":
            actions = model(obs[None])[0]
        elif algo == "ddpg":
            actions = model.pi(obs, apply_noise=False, compute_Q=False)[0]
        elif algo == "ars":
            actions = [model.getAction(obs.flatten())]
        elif algo == "cma-es":
            actions = [model.getAction(obs)]
        obs, rewards, dones, _ = envs.step(actions)

        if algo in ["deepq", "ddpg"]:
            if dones:
                obs = envs.reset()
            dones = [dones]
        n_done += sum(dones)
        if (n_done - last_n_done) > 1:
            last_n_done = n_done
            _, mean_reward = computeMeanReward(log_dir, n_done)
            print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))

    _, mean_reward = computeMeanReward(log_dir, n_done)
    print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))


if __name__ == '__main__':
    main()
