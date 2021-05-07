import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from rl_baselines.visualize import loadCsv, movingAverage, loadData
from srl_zoo.utils import printGreen, printYellow, printRed

# Init seaborn
sns.set()
# Style for the title
fontstyle = {'fontname': 'DejaVu Sans', 'fontsize': 16}

# Modified Colorbrewer Paired_12, you can use palettable to retrieve it
colors = [[166, 206, 227], [31, 120, 180], [178, 223, 138], [51, 160, 44], [251, 154, 153], [227, 26, 28],
          [253, 191, 111], [255, 127, 0], [202, 178, 214], [106, 61, 154], [143, 156, 212], [64, 57, 178], [255, 255, 153], [177, 89, 40],
          [10, 10, 10], [0, 0, 0]]
colors = [(r / 255, g / 255, b / 255) for (r, g, b) in colors]
lightcolors = colors[0::2]
darkcolors = colors[1::2]

# Default y-limits for the plot
# Kuka Arm
Y_LIM_SPARSE_REWARD = [0, 5]
# Mobile robot
Y_LIM_SPARSE_REWARD = [-3, 250]
# Relative: [-150, -50]
# Normal: [-70, -35]
Y_LIM_SHAPED_REWARD = [-150, -50]


def loadEpisodesData(folder):
    """
    :param folder: (str)
    :return: (numpy array, numpy array) or (None, None)
    """
    result, _ = loadCsv(folder)

    if len(result) == 0:
        return None, None

    y = np.array(result)[:, 1]
    x = np.arange(len(y))
    return x, y


def millions(x, pos):
    """
    formatter for matplotlib
    The two args are the value and tick position
    :param x: (float)
    :param pos: (int) tick position (not used here
    :return: (str)
    """
    return '{:.1f}M'.format(x * 1e-6)


def plotGatheredExperiments(folders, algo, y_limits, window=40, title="", min_num_x=-1,
                            timesteps=False, output_file="", no_display=False):
    """
    Compute mean and standard error for several experiments and plot the learning curve
    :param folders: ([str]) Log folders, where the monitor.csv are stored
    :param window: (int) Smoothing window
    :param algo: (str) name of the RL algo
    :param title: (str) plot title
    :param min_num_x: (int) Minimum number of episode/timesteps to keep an experiment (default: -1, no minimum)
    :param timesteps: (bool) Plot timesteps instead of episodes
    :param y_limits: ([float]) y-limits for the plot
    :param output_file: (str) Path to a file where the plot data will be saved
    :param no_display: (bool) Set to true, the plot won't be displayed (useful when only saving plot)
    """
    y_list = []
    x_list = []
    ok = False
    for folder in folders:
        if timesteps:
            x, y = loadData(folder, smooth=1, bin_size=100)
            if x is not None:
                x, y = np.array(x), np.array(y)
        else:
            x, y = loadEpisodesData(folder)

        if x is None or (min_num_x > 0 and y.shape[0] < min_num_x):
            printYellow("Skipping {}".format(folder))
            continue

        if y.shape[0] <= window:
            printYellow("Folder {}".format(folder))
            printYellow("Not enough episodes for current window size = {}".format(window))
            continue
        ok = True
        y = movingAverage(y, window)
        y_list.append(y)

        # Truncate x
        x = x[len(x) - len(y):]
        x_list.append(x)

    if not ok:
        printRed("Not enough data to plot anything with current config." +
                 " Consider decreasing --min-x")
        return

    lengths = list(map(len, x_list))
    min_x, max_x = np.min(lengths), np.max(lengths)

    print("Min x: {}".format(min_x))
    print("Max x: {}".format(max_x))

    for i in range(len(x_list)):
        x_list[i] = x_list[i][:min_x]
        y_list[i] = y_list[i][:min_x]

    x = np.array(x_list)[0]
    y = np.array(y_list)

    printGreen("{} Experiments".format(y.shape[0]))
    print("Min, Max rewards:", np.min(y), np.max(y))

    fig = plt.figure(title)
    # Compute mean for different seeds
    m = np.mean(y, axis=0)
    # Compute standard error
    s = np.squeeze(np.asarray(np.std(y, axis=0)))
    n = y.shape[0]
    plt.fill_between(x, m - s / np.sqrt(n), m + s / np.sqrt(n), color=lightcolors[0])
    plt.plot(x, m, color=darkcolors[0], label=algo, linewidth=1)

    if timesteps:
        formatter = FuncFormatter(millions)
        plt.xlabel('Number of Timesteps')
        fig.axes[0].xaxis.set_major_formatter(formatter)
    else:
        plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')

    plt.title(title, **fontstyle)
    plt.ylim(y_limits)

    plt.legend(framealpha=0.5, labelspacing=0.01, loc='lower right', fontsize=16)

    if output_file != "":
        printGreen("Saving aggregated data to {}.npz".format(output_file))
        np.savez(output_file, x=x, y=y)

    if not no_display:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trained agent")
    parser.add_argument('-i', '--log-dir', help='folder with the saved agent model', type=str, required=True)
    parser.add_argument('-o', '--output-file', help='Where to save the aggregated data', type=str, default="")
    parser.add_argument('--episode_window', type=int, default=40,
                        help='Episode window for moving average plot (default: 40)')
    parser.add_argument('--min-x', type=int, default=-1,
                        help='Minimum number of x-ticks to keep an experiment (default: -1, no minimum)')
    parser.add_argument('--y-lim', nargs=2, type=float, default=[-1, -1], help="limits for the y axis")
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Shape the reward (reward = - distance) instead of a sparse reward')
    parser.add_argument('--timesteps', action='store_true', default=False,
                        help='Plot timesteps instead of episodes')
    parser.add_argument('--no-display', action='store_true', default=False, help='Do not display plot')
    args = parser.parse_args()

    y_limits = args.y_lim
    if y_limits[0] == y_limits[1]:
        if args.shape_reward:
            y_limits = Y_LIM_SHAPED_REWARD
        else:
            y_limits = Y_LIM_SPARSE_REWARD
        print("Using default limits:", y_limits)

    # TODO: check that the parameters are the same between Experiments
    folders = []
    other = []
    train_args = {}
    for folder in os.listdir(args.log_dir):
        path = "{}/{}/".format(args.log_dir, folder)
        env_globals = json.load(open(path + "env_globals.json", 'r'))
        train_args = json.load(open(path + "args.json", 'r'))
        if train_args["shape_reward"] == args.shape_reward:
            folders.append(path)
        else:
            other.append(path)

    if len(folders) == 0 and len(other) == 0:
        printYellow("No experiment found. Is the folder path {} correct?".format(args.log_dir))
        exit()
    elif len(folders) == 0:
        printYellow("No experiments found with the given criterion. However {} experiments".format(len(other)) +
                    " where found {} reward shaping. ".format("without" if args.shape_reward else "with") +
                    "Did you mean {} the flag '--shape-reward'?".format("without" if args.shape_reward else "with"))
        exit()

    srl_model = train_args['srl_model'] if train_args['srl_model'] != "" else "raw pixels"
    if args.timesteps:
        title = srl_model + " [Timesteps]"
    else:
        title = srl_model + " [Episodes]"

    plotGatheredExperiments(folders, train_args['algo'], y_limits=y_limits, window=args.episode_window,
                            title=title, min_num_x=args.min_x, no_display=args.no_display,
                            timesteps=args.timesteps, output_file=args.output_file)
#python -m replay.aggregate_plots --log-dir logs/OmnirobotEnv-v0/srl_combination/ppo2/  --timesteps --min-x 1000 -o logs/plot/