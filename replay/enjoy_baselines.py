"""
Enjoy script for OpenAI Baselines
"""
import argparse
import json
import os
from datetime import datetime

import yaml
import numpy as np
import tensorflow as tf
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns

from rl_baselines import registered_rl, AlgoType
from rl_baselines.utils import createTensorflowSession, computeMeanReward, CustomVecNormalize, VecFrameStack, \
    CustomDummyVecEnv, WrapFrameStack
from srl_zoo.utils import printYellow, printGreen
from environments.utils import makeEnv


def fixStateDim(states):
    """
    Fix for plotting when state_dim < 3
    :param states: (numpy array or [float])
    :return: (numpy array)
    """
    states = np.array(states)
    state_dim = states.shape[1]
    if state_dim < 3:
        tmp = np.zeros((states.shape[0], 3))
        tmp[:, :state_dim] = states
        return tmp
    return states


def parseArguments():
    """

    :return: (Arguments)
    """
    parser = argparse.ArgumentParser(description="Load trained RL model")
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--num-cpu', help='Number of processes', type=int, default=1)
    parser.add_argument('--log-dir', help='folder with the saved agent model', type=str, required=True)
    parser.add_argument('--num-timesteps', type=int, default=int(1e4))
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render the environment (show the GUI)')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    parser.add_argument('--plotting', action='store_true', default=False,
                        help='display in the latent space the current observation.')
    return parser.parse_args()


def loadConfigAndSetup(load_args):
    """
    Get the training config and setup the parameters
    :param load_args: (Arguments)
    :return: (dict, str, str, dict, dict)
    """
    with open('config/srl_models.yaml', 'rb') as f:
        srl_models = yaml.load(f)

    algo_name = ""
    for algo in list(registered_rl.keys()):
        if algo in load_args.log_dir:
            algo_name = algo
            break
    algo_class, algo_type, _ = registered_rl[algo_name]
    if algo_type == AlgoType.Other:
        raise ValueError(algo_name + " is not supported for replay")
    printGreen("\n" + algo_name + "\n")

    load_path = "{}/{}_model.pkl".format(load_args.log_dir, algo_name)

    env_globals = json.load(open(load_args.log_dir + "kuka_env_globals.json", 'r'))
    train_args = json.load(open(load_args.log_dir + "args.json", 'r'))
    # choose the right paths for the environment
    assert train_args["env"] in srl_models, \
        "Error: environment '{}', is not defined in 'config/srl_models.yaml'".format(train_args["env"])
    srl_models = srl_models[train_args["env"]]

    env_kwargs = {
        "renders": load_args.render,
        "shape_reward": load_args.shape_reward,  # Reward sparse or shaped
        "action_joints": train_args["action_joints"],
        "is_discrete": not train_args["continuous_actions"],
        "random_target": train_args.get('relative', False),
    }

    # load it, if it was defined
    if "action_repeat" in env_globals:
        env_kwargs["action_repeat"] = env_globals['action_repeat']
    elif "ACTION_REPEAT" in env_globals:
        env_kwargs["action_repeat"] = env_globals['ACTION_REPEAT']

    # Remove up action
    if train_args["env"] == "Kuka2ButtonGymEnv-v0":
        env_kwargs["force_down"] = env_globals.get('force_down', env_globals.get('FORCE_DOWN', True))
    else:
        env_kwargs["force_down"] = env_globals.get('force_down', env_globals.get('FORCE_DOWN', False))

    if train_args["srl_model"] != "":
        train_args["policy"] = "mlp"
        path = srl_models.get(train_args["srl_model"])

        if train_args["srl_model"] == "ground_truth":
            env_kwargs["use_ground_truth"] = True
        elif train_args["srl_model"] == "joints":
            env_kwargs["use_joints"] = True
        elif train_args["srl_model"] == "joints_position":
            env_kwargs["use_ground_truth"] = True
            env_kwargs["use_joints"] = True
        elif path is not None:
            env_kwargs["use_srl"] = True
            env_kwargs["srl_model_path"] = srl_models['log_folder'] + path
        else:
            raise ValueError("Unsupported value for srl-model: {}".format(train_args["srl_model"]))

    return train_args, load_path, algo_name, algo_class, srl_models, env_kwargs


# TODO: fix this?
def createEnv(load_args, train_args, algo_name, env_kwargs, log_dir="/tmp/gym/test/"):
    """
    Create the Gym environment
    :param load_args: (Arguments)
    :param train_args: (dict)
    :param algo_name: (str)
    :param env_kwargs: (dict) The extra arguments for the environment
    :param log_dir: (str) Log dir for testing the agent
    :return: (str, SubprocVecEnv)
    """
    # Log dir for testing the agent
    log_dir += "{}/{}/".format(algo_name, datetime.now().strftime("%y-%m-%d_%Hh%M_%S"))
    os.makedirs(log_dir, exist_ok=True)

    if algo_name not in ["deepq", "ddpg"]:
        envs = SubprocVecEnv([makeEnv(train_args['env'], load_args.seed, i, log_dir, env_kwargs=env_kwargs)
                              for i in range(load_args.num_cpu)])
        envs = VecFrameStack(envs, train_args['num_stack'])
    else:
        if load_args.num_cpu > 1:
            printYellow(algo_name + " does not support multiprocessing, setting num-cpu=1")
        envs = CustomDummyVecEnv([makeEnv(train_args['env'], load_args.seed, 0, log_dir, env_kwargs=env_kwargs)])

    if train_args["srl_model"] != "":
        # special case where ars type v1 is not normalized
        if algo_name != "ars" or train_args['algo_type'] != "v1":
            envs = CustomVecNormalize(envs, training=False)
        # Temp fix for experiments where no running average were saved
        try:
            printGreen("Loading saved running average")
            envs.loadRunningAverage(load_args.log_dir)
        except FileNotFoundError:
            envs.training = True
            printYellow("Running Average files not found for CustomVecNormalize, switching to training mode")

    if algo_name in ["deepq", "ddpg"]:
        # Normalize only raw pixels
        normalize = train_args['srl_model'] == ""
        envs = WrapFrameStack(envs, train_args['num_stack'], normalize=normalize)

    return log_dir, envs


def main():
    load_args = parseArguments()
    train_args, load_path, algo_name, algo_class, srl_models, env_kwargs = loadConfigAndSetup(load_args)
    log_dir, envs = createEnv(load_args, train_args, algo_name, env_kwargs)

    assert (not load_args.plotting) or load_args.num_cpu == 1, "Error: cannot run plotting with more than 1 CPU"

    tf.reset_default_graph()
    set_global_seeds(load_args.seed)
    createTensorflowSession()

    printYellow("Compiling Policy function....")
    method = algo_class.load(load_path)

    dones = [False for _ in range(load_args.num_cpu)]
    obs = envs.reset()
    # print(obs.shape)

    # plotting init
    if load_args.plotting:
        plt.pause(0.1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        old_obs = []
        line, = ax.plot([], [], [], c=[1, 0, 0, 1], label="episode 0")
        point = ax.scatter([0], [0], [0], c=[1, 0, 0, 1])
        fig.legend()

        if train_args["srl_model"] in ["ground_truth", "supervised"]:
            ax.set_xlim([-0.4, 0.4])
            ax.set_ylim([-0.4, 0.4])
            ax.set_zlim([-0.2, 0.2])
            delta_obs = [envs.getOriginalObs()[0]]
        elif train_args["srl_model"] in ["vae", "autoencoder", "srl_priors"]:
            # we need to rebuild the PCA representation, in order to visualize correctly in 3D
            # load the saved representations
            path = srl_models['log_folder'] + "/".join(
                srl_models.get(train_args["srl_model"]).split("/")[:-1]) + "/image_to_state.json"
            X = np.array(list(json.load(open(path, 'r')).values()))

            X = fixStateDim(X)

            # train the PCA and et the limits
            pca = PCA(n_components=3)
            X_new = pca.fit_transform(X)
            ax.set_xlim([np.min(X_new[:, 0]) * 1.2, np.max(X_new[:, 0]) * 1.2])
            ax.set_ylim([np.min(X_new[:, 1]) * 1.2, np.max(X_new[:, 1]) * 1.2])
            ax.set_zlim([np.min(X_new[:, 2]) * 1.2, np.max(X_new[:, 2]) * 1.2])
            delta_obs = [pca.transform(fixStateDim([obs[0]]))[0]]
        else:
            assert False, "Error: srl_model {} not supported with plotting.".format(train_args["srl_model"])

    n_done = 0
    last_n_done = 0
    episode = 0
    for i in range(load_args.num_timesteps):
        actions = method.getAction(obs, dones)
        obs, rewards, dones, _ = envs.step(actions)

        # plotting
        if load_args.plotting:
            if train_args["srl_model"] in ["ground_truth", "supervised"]:
                ajusted_obs = envs.getOriginalObs()[0]
            elif train_args["srl_model"] in ["vae", "autoencoder", "srl_priors"]:
                ajusted_obs = pca.transform(fixStateDim([obs[0]]))[0]

            # create a new line, if the episode is finished
            if sum(dones) > 0:
                old_obs.append(np.array(delta_obs))
                line.set_c(sns.color_palette()[episode % len(sns.color_palette())])
                episode += 1
                line, = ax.plot([], [], [], c=[1, 0, 0, 1], label="episode " + str(episode))
                fig.legend()
                delta_obs = [ajusted_obs]
            else:
                delta_obs.append(ajusted_obs)

            coor_plt = fixStateDim(np.array(delta_obs))

            # updating the 3d vertices for the line and the dot drawing, to avoid redrawing the entire image
            line._verts3d = (coor_plt[:, 0], coor_plt[:, 1], coor_plt[:, 2])
            point._offsets3d = ([coor_plt[-1, 0]], [coor_plt[-1, 1]], [coor_plt[-1, 2]])

            # Draw every 5 frames to avoid UI freezing
            if i % 5 == 0:
                fig.canvas.draw()
                plt.pause(0.000001)

        # TODO: fix this?
        if algo_name in ["deepq", "ddpg"]:
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
