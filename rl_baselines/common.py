import os
import argparse
from datetime import datetime

import yaml
from visdom import Visdom

from pytorch_agents.visualize import visdom_plot, episode_plot
import rl_baselines.deepq as deepq
import rl_baselines.acer as acer
import rl_baselines.random_agent as random_agent

VISDOM_PORT = 8097
LOG_INTERVAL = 10
LOG_DIR = ""
ALGO = ""
ENV_NAME = ""
PLOT_TITLE = "Raw Pixels"
EPISODE_WINDOW = 20  # For plotting moving average
viz = None
n_steps = 0
SAVE_INTERVAL = 500  # Save RL model every 500 steps

win, win_smooth, win_episodes = None, None, None

# LOAD SRL models list
with open('config/srl_models.yaml', 'rb') as f:
    models = yaml.load(f)


def configureEnvAndLogFolder(args, kuka_env):
    """
    :param args: (ArgumentParser object)
    :return: (ArgumentParser object)
    """
    global PLOT_TITLE, LOG_DIR

    if args.srl_model != "":
        PLOT_TITLE = args.srl_model
        path = models.get(args.srl_model)
        args.log_dir += args.srl_model + "/"

        if args.srl_model == "ground_truth":
            kuka_env.USE_GROUND_TRUTH = True
            PLOT_TITLE = "Ground Truth"
        elif path is not None:
            kuka_env.USE_SRL = True
            kuka_env.SRL_MODEL_PATH = models['log_folder'] + path
        else:
            raise ValueError("Unsupported value for srl-model: {}".format(args.srl_model))

    else:
        args.log_dir += "raw_pixels/"

    # Add date + current time
    args.log_dir += "{}/{}/".format(ALGO, datetime.now().strftime("%d-%m-%y_%Hh%M_%S"))
    LOG_DIR = args.log_dir

    os.makedirs(args.log_dir, exist_ok=True)

    return args


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN) or after n steps (see ACER)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global win, win_smooth, win_episodes, n_steps, viz
    if viz is None:
        viz = Visdom(port=VISDOM_PORT)

    # HACK to save RL model
    # TODO: check that the model has improved
    if (n_steps + 1) % SAVE_INTERVAL == 0:
        if ALGO == "deepq":
            _locals['act'].save(LOG_DIR + "deepq_model.pkl")
        elif ALGO == "acer":
            _locals['model'].save(LOG_DIR + "acer_model.pkl")

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
    parser.add_argument('--algo', default='deepq', choices=['acer', 'deepq', 'random'],
                        help='OpenAI baseline to use')
    parser.add_argument('--env', help='environment ID', default='KukaButtonGymEnv-v0')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs and model (default: /tmp/gym)')
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--srl-model', type=str, default='',
                        choices=["autoencoder", "ground_truth", "srl_priors", "supervised"],
                        help='SRL model to use')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 1)')
    parser.add_argument('--action-repeat', type=int, default=1,
                        help='number of times an action will be repeated (default: 1)')
    parser.add_argument('--port', type=int, default=8097,
                        help='visdom server port (default: 8097)')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')

    # Ignore unknown args for now
    args, unknown = parser.parse_known_args()

    ENV_NAME = args.env
    ALGO = args.algo
    VISDOM_PORT = args.port
    if args.no_vis:
        vis = False

    if args.algo == "deepq":
        algo = deepq
        LOG_INTERVAL = 100
    elif args.algo == "acer":
        algo = acer
        LOG_INTERVAL = 1
        SAVE_INTERVAL = 1
    elif args.algo == "random":
        algo = random_agent
        LOG_INTERVAL = 100

    print("\nAgent = {} \n".format(args.algo))

    algo.kuka_env.ACTION_REPEAT = args.action_repeat

    parser = algo.customArguments(parser)
    args = parser.parse_args()
    args = configureEnvAndLogFolder(args, algo.kuka_env)
    algo.main(args, callback)

if __name__ == '__main__':
    main()
