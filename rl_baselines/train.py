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

import rl_baselines.a2c as a2c
import rl_baselines.acer as acer
import rl_baselines.deepq as deepq
import rl_baselines.ppo2 as ppo2
import rl_baselines.random_agent as random_agent
import rl_baselines.random_search as random_search
import rl_baselines.ddpg as ddpg
from pytorch_agents.visualize import visdom_plot, episode_plot
from rl_baselines.utils import filterJSONSerializableObjects
from rl_baselines.utils import computeMeanReward
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

# LOAD SRL models list
with open('config/srl_models.yaml', 'rb') as f:
    models = yaml.load(f)


def saveEnvParams(kuka_env):
    """
    :param kuka_env: (kuka_env module)
    """
    params = filterJSONSerializableObjects(kuka_env.getGlobals())
    with open(LOG_DIR + "kuka_env_globals.json", "w") as f:
        json.dump(params, f)


def configureEnvAndLogFolder(args, kuka_env):
    """
    :param args: (ArgumentParser object)
    :param kuka_env: (kuka_env module)
    :return: (ArgumentParser object)
    """
    global PLOT_TITLE, LOG_DIR
    # Reward sparse or shaped
    kuka_env.SHAPE_REWARD = args.shape_reward
    kuka_env.ACTION_JOINTS = args.action_joints

    if args.srl_model != "":
        PLOT_TITLE = args.srl_model
        path = models.get(args.srl_model)
        args.log_dir += args.srl_model + "/"

        if args.srl_model == "ground_truth":
            kuka_env.USE_GROUND_TRUTH = True
            PLOT_TITLE = "Ground Truth"
        elif args.srl_model == "joints":
            kuka_env.USE_JOINTS = True
            PLOT_TITLE = "Joints"
        elif args.srl_model == "joints_position":
            kuka_env.USE_GROUND_TRUTH = True
            kuka_env.USE_JOINTS = True
            PLOT_TITLE = "Joints and position"
        elif path is not None:
            kuka_env.USE_SRL = True
            kuka_env.SRL_MODEL_PATH = models['log_folder'] + path
        else:
            raise ValueError("Unsupported value for srl-model: {}".format(args.srl_model))

    else:
        args.log_dir += "raw_pixels/"

    # Add date + current time
    args.log_dir += "{}/{}/".format(ALGO, datetime.now().strftime("%y-%m-%d_%Hh%M_%S"))
    LOG_DIR = args.log_dir

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
        ok, mean_reward = computeMeanReward(LOG_DIR, N_EPISODES_EVAL)
        if ok:
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
        else:
            # Not enough episode
            mean_reward = -10000

        # Save Best model
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            printGreen("Saving new best model")
            if ALGO == "deepq":
                _locals['act'].save(LOG_DIR + "deepq_model.pkl")
            elif ALGO == "ddpg":
                _locals['agent'].save(LOG_DIR + "ddpg_model.pkl")
            elif ALGO in ["acer", "a2c", "ppo2"]:
                _locals['model'].save(LOG_DIR + ALGO + "_model.pkl")
            elif "pytorch" in ALGO:
                # Bring back the weights to the cpu
                if _globals['args'].cuda:
                    _locals['actor_critic'].cpu()
                _globals['torch'].save(_locals['actor_critic'].state_dict(), "{}/{}_model.pth".format(LOG_DIR, ALGO))
                if _globals['args'].cuda:
                    _locals['actor_critic'].cuda()

    # Plots in visdom
    if viz and (n_steps + 1) % LOG_INTERVAL == 0:
        win = visdom_plot(viz, win, LOG_DIR, ENV_NAME, ALGO, bin_size=1, smooth=0, title=PLOT_TITLE)
        win_smooth = visdom_plot(viz, win_smooth, LOG_DIR, ENV_NAME, ALGO, title=PLOT_TITLE + " smoothed")
        win_episodes = episode_plot(viz, win_episodes, LOG_DIR, ENV_NAME, ALGO, window=EPISODE_WINDOW,
                                    title=PLOT_TITLE + " [Episodes]")
    n_steps += 1
    return False


def main():
    global ENV_NAME, ALGO, LOG_INTERVAL, VISDOM_PORT, viz, SAVE_INTERVAL
    parser = argparse.ArgumentParser(description="OpenAI RL Baselines")
    parser.add_argument('--algo', default='deepq', choices=['acer', 'deepq', 'a2c', 'ppo2', 'random_search', 'random_agent', 'ddpg'],
                        help='OpenAI baseline to use')
    parser.add_argument('--env', help='environment ID', default='KukaButtonGymEnv-v0')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs and model (default: /tmp/gym)')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--srl-model', type=str, default='',
                        choices=["autoencoder", "ground_truth", "srl_priors", "supervised", "pca", "joints", "joints_position"],
                        help='SRL model to use')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4)')
    parser.add_argument('--action-repeat', type=int, default=1,
                        help='number of times an action will be repeated (default: 1)')
    parser.add_argument('--port', type=int, default=8097,
                        help='visdom server port (default: 8097)')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    parser.add_argument('-c','--continuous-actions', action='store_true', default=False)
    parser.add_argument('-joints','--action-joints', help='set actions to the joints of the arm directly, instead of inverse kinematicsn', action='store_true', default=False)

    # Ignore unknown args for now
    args, unknown = parser.parse_known_args()

    ENV_NAME = args.env
    ALGO = args.algo
    VISDOM_PORT = args.port
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
    elif args.algo == "random_search":
        algo = random_search
    elif args.algo == "ddpg":
        algo = ddpg
        assert args.continuous_actions, "DDPG only works with '--continuous-actions' (or '-c')"

    if args.continuous_actions and (args.algo in ['acer', 'deepq', 'a2c', 'random_search']):
        raise ValueError(args.algo + " does not support continuous actions")

    algo.kuka_env.IS_DISCRETE = not args.continuous_actions

    printGreen("\nAgent = {} \n".format(args.algo))

    algo.kuka_env.ACTION_REPEAT = args.action_repeat

    parser = algo.customArguments(parser)
    args = parser.parse_args()
    args = configureEnvAndLogFolder(args, algo.kuka_env)
    args_dict = filterJSONSerializableObjects(vars(args))
    # Save args
    with open(LOG_DIR + "args.json", "w") as f:
        json.dump(args_dict, f)

    # Print Variables
    printYellow("Arguments:")
    pprint(args_dict)
    printYellow("Kuka Env Globals:")
    pprint(filterJSONSerializableObjects(algo.kuka_env.getGlobals()))
    # Save kuka env params
    saveEnvParams(algo.kuka_env)
    # Seed tensorflow, python and numpy random generator
    set_global_seeds(args.seed)
    algo.main(args, callback)


if __name__ == '__main__':
    main()
