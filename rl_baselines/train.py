"""
Train script for openAI RL Baselines
"""
import argparse
import json
import os
from datetime import datetime
from pprint import pprint

import yaml
from baselines.common import set_global_seeds
from visdom import Visdom
import tensorflow as tf
import gym.envs.registration

import rl_baselines.a2c as a2c
import rl_baselines.acer as acer
import rl_baselines.ddpg as ddpg
import rl_baselines.deepq as deepq
import rl_baselines.ppo2 as ppo2
import rl_baselines.random_agent as random_agent
import rl_baselines.ars as ars
import rl_baselines.cma_es as cma_es
from rl_baselines.utils import computeMeanReward
from rl_baselines.utils import filterJSONSerializableObjects
from rl_baselines.visualize import timestepsPlot, episodePlot
from srl_priors.utils import printGreen, printYellow

VISDOM_PORT = 8097
LOG_INTERVAL = 100
LOG_DIR = ""
ALGO = ""
ENV_NAME = ""
PLOT_TITLE = "Raw Pixels"
EPISODE_WINDOW = 40  # For plotting moving average
viz = None
n_steps = 0
SAVE_INTERVAL = 500  # Save RL model every 500 steps
N_EPISODES_EVAL = 100  # Evaluate the performance on the last 100 episodes
params_saved = False
best_mean_reward = -10000

win, win_smooth, win_episodes = None, None, None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow

# LOAD SRL models list
with open('config/srl_models.yaml', 'rb') as f:
    models = yaml.load(f)


def saveEnvParams(kuka_env, env_kwargs):
    """
    :param kuka_env: (kuka_env module)
    :param env_kwargs: (dict) The extra arguments for the environment
    """
    params = filterJSONSerializableObjects({**kuka_env.getGlobals(), **env_kwargs})
    with open(LOG_DIR + "kuka_env_globals.json", "w") as f:
        json.dump(params, f)


def configureEnvAndLogFolder(args, env_kwargs):
    """
    :param args: (ArgumentParser object)
    :param env_kwargs: (dict) The extra arguments for the environment
    :return: (ArgumentParser object, dict)
    """
    global PLOT_TITLE, LOG_DIR
    # Reward sparse or shaped
    env_kwargs["shape_reward"] = args.shape_reward
    # Actions in joint space or relative position space
    env_kwargs["action_joints"] = args.action_joints
    args.log_dir += args.env + "/"

    if args.srl_model != "":
        PLOT_TITLE = args.srl_model
        path = models.get(args.srl_model)
        args.log_dir += args.srl_model + "/"

        if args.srl_model == "ground_truth":
            env_kwargs["use_ground_truth"] = True
            PLOT_TITLE = "Ground Truth"
        elif args.srl_model == "joints":
            # Observations in joint space
            env_kwargs["use_joints"] = True
            PLOT_TITLE = "Joints"
        elif args.srl_model == "joints_position":
            # Observations in joint and position space
            env_kwargs["use_ground_truth"] = True
            env_kwargs["use_joints"] = True
            PLOT_TITLE = "Joints and position"
        elif path is not None:
            env_kwargs["use_srl"] = True
            env_kwargs["srl_model_path"] = models['log_folder'] + path
        else:
            raise ValueError("Unsupported value for srl-model: {}".format(args.srl_model))

    else:
        args.log_dir += "raw_pixels/"

    # Add date + current time
    args.log_dir += "{}/{}/".format(ALGO, datetime.now().strftime("%y-%m-%d_%Hh%M_%S"))
    LOG_DIR = args.log_dir
    # TODO: wait one second if the folder exist to avoid overwritting logs
    os.makedirs(args.log_dir, exist_ok=True)

    return args


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global win, win_smooth, win_episodes, n_steps, viz, params_saved, best_mean_reward
    # Create vizdom object only if needed
    if viz is None:
        viz = Visdom(port=VISDOM_PORT)

    is_es = ALGO in ['ars', 'cma-es']

    # Save RL agent parameters
    if not params_saved:
        # Filter locals
        params = filterJSONSerializableObjects(_locals)
        with open(LOG_DIR + "rl_locals.json", "w") as f:
            json.dump(params, f)
        params_saved = True

    # Save the RL model if it has improved
    if (n_steps + 1) % SAVE_INTERVAL == 0:
        # Evaluate network performance
        ok, mean_reward = computeMeanReward(LOG_DIR, N_EPISODES_EVAL, is_es=is_es)
        if ok:
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
        else:
            # Not enough episode
            mean_reward = -10000

        # Save Best model
        if mean_reward > best_mean_reward:
            # Try saving the running average (only valid for mlp policy)
            try:
                _locals['env'].saveRunningAverage(LOG_DIR)
            except AttributeError:
                pass

            best_mean_reward = mean_reward
            printGreen("Saving new best model")
            if ALGO == "deepq":
                _locals['act'].save(LOG_DIR + "deepq_model.pkl")
            elif ALGO == "ddpg":
                _locals['agent'].save(LOG_DIR + "ddpg_model.pkl")
            elif ALGO in ["acer", "a2c", "ppo2"]:
                _locals['model'].save(LOG_DIR + ALGO + "_model.pkl")
            elif ALGO in ['ars', 'cma-es']:
                _locals['self'].save(LOG_DIR + ALGO + "_model.pkl")


    # Plots in visdom
    if viz and (n_steps + 1) % LOG_INTERVAL == 0:
        win = timestepsPlot(viz, win, LOG_DIR, ENV_NAME, ALGO, bin_size=1, smooth=0, title=PLOT_TITLE, is_es=is_es)
        win_smooth = timestepsPlot(viz, win_smooth, LOG_DIR, ENV_NAME, ALGO, title=PLOT_TITLE + " smoothed", is_es=is_es)
        win_episodes = episodePlot(viz, win_episodes, LOG_DIR, ENV_NAME, ALGO, window=EPISODE_WINDOW,
                                   title=PLOT_TITLE + " [Episodes]", is_es=is_es)
    n_steps += 1
    return False


def main():
    global ENV_NAME, ALGO, LOG_INTERVAL, VISDOM_PORT, viz, SAVE_INTERVAL, EPISODE_WINDOW
    parser = argparse.ArgumentParser(description="OpenAI RL Baselines")
    parser.add_argument('--algo', default='ppo2',
                        choices=['acer', 'deepq', 'a2c', 'ppo2', 'random_agent', 'ddpg', 'ars', 'cma-es'],
                        help='OpenAI baseline to use', type=str)
    parser.add_argument('--env', type=str, help='environment ID', default='KukaButtonGymEnv-v0')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--episode_window', type=int, default=40,
                        help='Episode window for moving average plot (default: 40)')
    parser.add_argument('--log-dir', default='/tmp/gym/', type=str,
                        help='directory to save agent logs and model (default: /tmp/gym)')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--srl-model', type=str, default='',
                        choices=["autoencoder", "ground_truth", "srl_priors", "supervised", "pca", "vae", "joints", "joints_position"],
                        help='SRL model to use')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 1)')
    parser.add_argument('--action-repeat', type=int, default=1,
                        help='number of times an action will be repeated (default: 1)')
    parser.add_argument('--port', type=int, default=8097,
                        help='visdom server port (default: 8097)')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    parser.add_argument('-c', '--continuous-actions', action='store_true', default=False)
    parser.add_argument('-joints', '--action-joints',
                        help='set actions to the joints of the arm directly, instead of inverse kinematics',
                        action='store_true', default=False)
    parser.add_argument('-r', '--relative', action='store_true', default=False, help='Set the button to a random position')

    # Ignore unknown args for now
    args, unknown = parser.parse_known_args()
    env_kwargs = {}

    # Sanity check
    assert args.env in gym.envs.registration.registry.env_specs, \
        "Error: could not find the environment {}".format(args.env)
    assert args.episode_window >= 1, "Error: --episode_window cannot be less than 1"
    assert args.num_timesteps >= 1, "Error: --num-timesteps cannot be less than 1"
    assert args.num_stack >= 1, "Error: --num-stack cannot be less than 1"
    assert args.action_repeat >= 1, "Error: --action-repeat cannot be less than 1"
    assert args.port >= 0 and args.port < 65535, \
        "Error: invalid visdom port number {}, port number must be an unsigned 16bit number [0,65535].".format(args.port)

    ENV_NAME = args.env
    ALGO = args.algo
    VISDOM_PORT = args.port
    EPISODE_WINDOW = args.episode_window

    if args.no_vis:
        viz = False

    if args.algo == "deepq":
        algo = deepq
    elif args.algo == "acer":
        algo = acer
        # callback is not called after each steps
        # so we need to reduce log and save interval
        LOG_INTERVAL = 1
        SAVE_INTERVAL = 20
    elif args.algo == "a2c":
        algo = a2c
    elif args.algo == "ppo2":
        algo = ppo2
        LOG_INTERVAL = 10
        SAVE_INTERVAL = 10
    elif args.algo == "random_agent":
        algo = random_agent
    elif args.algo == "ddpg":
        algo = ddpg
        assert args.continuous_actions, "DDPG only works with '--continuous-actions' (or '-c')"
    elif args.algo == "ars":
        algo = ars
    elif args.algo == "cma-es":
        algo = cma_es

    if args.continuous_actions and (args.algo in ['acer', 'deepq', 'a2c', 'random_search']):
        raise ValueError(args.algo + " does not support continuous actions")

    env_kwargs["is_discrete"] = not args.continuous_actions

    printGreen("\nAgent = {} \n".format(args.algo))

    env_kwargs["action_repeat"] = args.action_repeat
    # Random init position for button
    env_kwargs["button_random"] = args.relative
    # Allow up action
    # env_kwargs["force_down"] = False

    parser = algo.customArguments(parser)
    args = parser.parse_args()

    args, env_kwargs = configureEnvAndLogFolder(args, env_kwargs)
    args_dict = filterJSONSerializableObjects(vars(args))
    # Save args
    with open(LOG_DIR + "args.json", "w") as f:
        json.dump(args_dict, f)

    # Print Variables
    printYellow("Arguments:")
    pprint(args_dict)
    printYellow("Kuka Env Globals:")
    pprint(filterJSONSerializableObjects({**algo.kuka_env.getGlobals(), **env_kwargs}))
    # Save kuka env params
    saveEnvParams(algo.kuka_env, env_kwargs)
    # Seed tensorflow, python and numpy random generator
    set_global_seeds(args.seed)
    # Augment the number of timesteps (when using mutliprocessing this number is not reached)
    args.num_timesteps = int(1.1 * args.num_timesteps)
    # Train the agent
    algo.main(args, callback, env_kwargs=env_kwargs) 


if __name__ == '__main__':
    main()
