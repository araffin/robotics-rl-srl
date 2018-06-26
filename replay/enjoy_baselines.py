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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns

from rl_baselines import AlgoType
from rl_baselines.registry import registered_rl
from rl_baselines.utils import createTensorflowSession, computeMeanReward, WrapFrameStack, softmax
from srl_zoo.utils import printYellow, printGreen
# has to be imported here, as otherwise it will cause loading of undefined functions
from environments import PlottingType
from environments.registry import registered_env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow


def fixStateDim(states):
    """
    Fix for plotting when state_dim < 2
    :param states: (numpy array or [float])
    :return: (numpy array)
    """
    states = np.array(states)
    state_dim = states.shape[1]
    if state_dim < 2:
        tmp = np.zeros((states.shape[0], 2))
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
    parser.add_argument('--action-proba', action='store_true', default=False,
                        help='display the probability of actions')
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
    if algo_type == AlgoType.OTHER:
        raise ValueError(algo_name + " is not supported for replay")
    printGreen("\n" + algo_name + "\n")

    load_path = "{}/{}_model.pkl".format(load_args.log_dir, algo_name)

    env_globals = json.load(open(load_args.log_dir + "env_globals.json", 'r'))
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
        "srl_model": train_args["srl_model"]
    }

    # load it, if it was defined
    if "action_repeat" in env_globals:
        env_kwargs["action_repeat"] = env_globals['action_repeat']

    # Remove up action
    if train_args["env"] == "Kuka2ButtonGymEnv-v0":
        env_kwargs["force_down"] = env_globals.get('force_down', True)
    else:
        env_kwargs["force_down"] = env_globals.get('force_down', False)

    if train_args["srl_model"] != "":
        train_args["policy"] = "mlp"
        path = srl_models.get(train_args["srl_model"])

        if path is not None:
            env_kwargs["use_srl"] = True
            env_kwargs["srl_model_path"] = srl_models['log_folder'] + path

    return train_args, load_path, algo_name, algo_class, srl_models, env_kwargs


def createEnv(load_args, train_args, algo_name, algo_class, env_kwargs, log_dir="/tmp/gym/test/"):
    """
    Create the Gym environment
    :param load_args: (Arguments)
    :param train_args: (dict)
    :param algo_name: (str)
    :param algo_class: (Class) a BaseRLObject subclass
    :param env_kwargs: (dict) The extra arguments for the environment
    :param log_dir: (str) Log dir for testing the agent
    :return: (str, SubprocVecEnv)
    """
    # Log dir for testing the agent
    log_dir += "{}/{}/".format(algo_name, datetime.now().strftime("%y-%m-%d_%Hh%M_%S"))
    os.makedirs(log_dir, exist_ok=True)

    args = {
        "env": train_args['env'],
        "seed": load_args.seed,
        "num_cpu": load_args.num_cpu,
        "num_stack": train_args["num_stack"],
        "srl_model": train_args["srl_model"],
        "algo_type": train_args.get('algo_type', None),
        "log_dir": log_dir
    }
    algo_args = type('attrib_dict', (), args)()  # anonymous class so the dict looks like Arguments object
    envs = algo_class.makeEnv(algo_args, env_kwargs=env_kwargs, load_path_normalise=load_args.log_dir)

    return log_dir, envs, algo_args


def main():
    load_args = parseArguments()
    train_args, load_path, algo_name, algo_class, srl_models, env_kwargs = loadConfigAndSetup(load_args)
    log_dir, envs, algo_args = createEnv(load_args, train_args, algo_name, algo_class, env_kwargs)

    assert (not load_args.plotting and not load_args.action_proba)\
        or load_args.num_cpu == 1, "Error: cannot run plotting with more than 1 CPU"

    tf.reset_default_graph()
    set_global_seeds(load_args.seed)
    createTensorflowSession()

    printYellow("Compiling Policy function....")
    method = algo_class.load(load_path, args=algo_args)

    dones = [False for _ in range(load_args.num_cpu)]
    # HACK: check for custom vec env
    using_custom_vec_env = isinstance(envs, WrapFrameStack)

    obs = envs.reset()
    if using_custom_vec_env:
        obs = [obs]
    # print(obs.shape)

    # plotting init
    if load_args.plotting:
        plt.pause(0.1)
        fig = plt.figure()
        old_obs = []
        if registered_env[train_args["env"]][2] == PlottingType.PLOT_3D:
            ax = fig.add_subplot(111, projection='3d')
            line, = ax.plot([], [], [], c=[1, 0, 0, 1], label="episode 0")
            point = ax.scatter([0], [0], [0], c=[1, 0, 0, 1])
            min_zone = [+np.inf, +np.inf, +np.inf]
            max_zone = [-np.inf, -np.inf, -np.inf]
            amplitude = [0, 0, 0]
        else:
            ax = fig.add_subplot(111)
            line, = ax.plot([], [], c=[1, 0, 0, 1], label="episode 0")
            point = ax.scatter([0], [0], c=[1, 0, 0, 1])
            min_zone = [+np.inf, +np.inf]
            max_zone = [-np.inf, -np.inf]
            amplitude = [0, 0]
        fig.legend()

        if train_args["srl_model"] in ["ground_truth", "supervised"]:
            delta_obs = [envs.getOriginalObs()[0]]
        else:
            # we need to rebuild the PCA representation, in order to visualize correctly in 3D
            # load the saved representations
            path = srl_models['log_folder'] + "/".join(
                srl_models.get(train_args["srl_model"]).split("/")[:-1]) + "/image_to_state.json"
            X = np.array(list(json.load(open(path, 'r')).values()))

            X = fixStateDim(X)

            # train the PCA and et the limits
            if registered_env[train_args["env"]][2] == PlottingType.PLOT_3D:
                pca = PCA(n_components=3)
            else:
                pca = PCA(n_components=2)
            X_new = pca.fit_transform(X)
            delta_obs = [pca.transform(fixStateDim([obs[0]]))[0]]
        plt.pause(0.00001)

    if load_args.action_proba and hasattr(method, "getActionProba"):
        fig_prob = plt.figure()
        ax_prob = fig_prob.add_subplot(111)
        old_obs = []
        if train_args["continuous_actions"]:
            ax_prob.set_ylim(np.min(envs.action_space.low), np.max(envs.action_space.high))
            bar = ax_prob.bar(np.arange(np.prod(envs.action_space.shape)),
                              np.array([0] * np.prod(envs.action_space.shape)),
                              color=plt.get_cmap('viridis')(int(1 / np.prod(envs.action_space.shape) * 255)))
        else:
            ax_prob.set_ylim(0, 1)
            bar = ax_prob.bar(np.arange(envs.action_space.n), np.array([0] * envs.action_space.n),
                              color=plt.get_cmap('viridis')(int(1 / envs.action_space.n * 255)))
        plt.pause(1)
        background_prob = fig_prob.canvas.copy_from_bbox(ax_prob.bbox)

    n_done = 0
    last_n_done = 0
    episode = 0
    for i in range(load_args.num_timesteps):
        actions = method.getAction(obs, dones)
        obs, rewards, dones, _ = envs.step(actions)
        if using_custom_vec_env:
            obs = [obs]

        # plotting
        if load_args.plotting:
            if train_args["srl_model"] in ["ground_truth", "supervised"]:
                ajusted_obs = envs.getOriginalObs()[0]
            else:
                ajusted_obs = pca.transform(fixStateDim([obs[0]]))[0]

            # create a new line, if the episode is finished
            if np.sum(dones) > 0:
                old_obs.append(np.array(delta_obs))
                line.set_c(sns.color_palette()[episode % len(sns.color_palette())])
                episode += 1
                if registered_env[train_args["env"]][2] == PlottingType.PLOT_3D:
                    line, = ax.plot([], [], [], c=[1, 0, 0, 1], label="episode " + str(episode))
                else:
                    line, = ax.plot([], [], c=[1, 0, 0, 1], label="episode " + str(episode))
                fig.legend()
                delta_obs = [ajusted_obs]
            else:
                delta_obs.append(ajusted_obs)

            coor_plt = fixStateDim(np.array(delta_obs))[1:]
            unstack_val = coor_plt.shape[1] // train_args.get("num_stack", 1)
            coor_plt = coor_plt[:, -unstack_val:]

            # updating the 3d vertices for the line and the dot drawing, to avoid redrawing the entire image
            if registered_env[train_args["env"]][2] == PlottingType.PLOT_3D:
                line._verts3d = (coor_plt[:, 0], coor_plt[:, 1], coor_plt[:, 2])
                point._offsets3d = (coor_plt[-1:, 0], coor_plt[-1:, 1], coor_plt[-1:, 2])
                if coor_plt.shape[0] > 0:
                    min_zone = np.minimum(np.amin(coor_plt, axis=0), min_zone)
                    max_zone = np.maximum(np.amax(coor_plt, axis=0), max_zone)
                    amplitude = max_zone - min_zone + 1e-10
                ax.set_xlim(min_zone[0] - abs(amplitude[0] * 0.2), max_zone[0] + abs(amplitude[0] * 0.2))
                ax.set_ylim(min_zone[1] - abs(amplitude[1] * 0.2), max_zone[1] + abs(amplitude[1] * 0.2))
                ax.set_zlim(min_zone[2] - abs(amplitude[2] * 0.2), max_zone[2] + abs(amplitude[2] * 0.2))
            else:
                line.set_xdata(coor_plt[:, 0])
                line.set_ydata(coor_plt[:, 1])
                point._offsets = coor_plt[-1:, :]
                if coor_plt.shape[0] > 0:
                    min_zone = np.minimum(np.amin(coor_plt, axis=0), min_zone)
                    max_zone = np.maximum(np.amax(coor_plt, axis=0), max_zone)
                    amplitude = max_zone - min_zone + 1e-10
                ax.set_xlim(min_zone[0] - abs(amplitude[0] * 0.2), max_zone[0] + abs(amplitude[0] * 0.2))
                ax.set_ylim(min_zone[1] - abs(amplitude[1] * 0.2), max_zone[1] + abs(amplitude[1] * 0.2))

            # Draw every 5 frames to avoid UI freezing
            if i % 5 == 0:
                fig.canvas.draw()
                plt.pause(0.000001)

        if load_args.action_proba and hasattr(method, "getActionProba"):
            pi = method.getActionProba(obs, dones)
            fig_prob.canvas.restore_region(background_prob)
            for act, rect in enumerate(bar):
                if train_args["continuous_actions"]:
                    rect.set_height(pi[0][act])
                    color_val = np.abs(pi[0][act]) / max(np.max(envs.action_space.high),
                                                         np.max(np.abs(envs.action_space.low)))
                else:
                    rect.set_height(softmax(pi[0])[act])
                    color_val = softmax(pi[0])[act]
                rect.set_color(plt.get_cmap('viridis')(int(color_val * 255)))
                ax_prob.draw_artist(rect)
            fig_prob.canvas.blit(ax_prob.bbox)

            # if i % 5 == 0:
            #     plt.pause(0.000001)

        if using_custom_vec_env:
            if dones:
                obs = [envs.reset()]

        n_done += np.sum(dones)
        if (n_done - last_n_done) > 1:
            last_n_done = n_done
            _, mean_reward = computeMeanReward(log_dir, n_done)
            print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))

    _, mean_reward = computeMeanReward(log_dir, n_done)
    print("{} episodes - Mean reward: {:.2f}".format(n_done, mean_reward))


if __name__ == '__main__':
    main()
