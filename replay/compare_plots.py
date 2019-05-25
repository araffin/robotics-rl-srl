import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib import rc

from replay.aggregate_plots import lightcolors, darkcolors, Y_LIM_SHAPED_REWARD, Y_LIM_SPARSE_REWARD, millions
from srl_zoo.utils import printGreen, printRed

# Init seaborn
sns.set()
# Style for the title
fontstyle = {'fontname': 'DejaVu Sans', 'fontsize': 22, 'fontweight': 'bold'}
rc('font', weight='bold')

def comparePlots(path, plots, y_limits, title="Learning Curve",
                 timesteps=False, truncate_x=-1, no_display=False):
    """
    :param path: (str) path to the folder where the plots are stored
    :param plots: ([str]) List of saved plots as npz file
    :param y_limits: ([float]) y-limits for the plot
    :param title: (str) plot title
    :param timesteps: (bool) Plot timesteps instead of episodes
    :param truncate_x: (int) Truncate the experiments after n ticks on the x-axis
    :param no_display: (bool) Set to true, the plot won't be displayed (useful when only saving plot)
    """
    y_list = []
    x_list = []
    for plot in plots:
        saved_plot = np.load('{}/{}'.format(path, plot))
        x_list.append(saved_plot['x'])
        y_list.append(saved_plot['y'])

    lengths = list(map(len, x_list))
    min_x, max_x = np.min(lengths), np.max(lengths)

    print("Min x: {}".format(min_x))
    print("Max x: {}".format(max_x))

    if truncate_x > 0:
        min_x = min(truncate_x, min_x)
    print("Truncating the x-axis at {}".format(min_x))

    x = np.array(x_list[0][:min_x])

    printGreen("{} Experiments".format(len(y_list)))
    # print("Min, Max rewards:", np.min(y), np.max(y))

    fig = plt.figure(title)
    for i in range(len(y_list)):
        label = plots[i].split('.npz')[0]
        y = y_list[i][:, :min_x]
        print('{}: {} experiments'.format(label, len(y)))
        # Compute mean for different seeds
        m = np.mean(y, axis=0)
        # Compute standard error
        s = np.squeeze(np.asarray(np.std(y, axis=0)))
        n = y.shape[0]
        plt.fill_between(x, m - s / np.sqrt(n), m + s / np.sqrt(n), color=lightcolors[i % len(lightcolors)], alpha=0.5)
        plt.plot(x, m, color=darkcolors[i % len(darkcolors)], label=label, linewidth=2)

    if timesteps:
        formatter = FuncFormatter(millions)
        plt.xlabel('Number of Timesteps', fontsize=20, fontweight='bold')
        fig.axes[0].xaxis.set_major_formatter(formatter)
    else:
        plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards', fontsize=20, fontweight='bold')

    plt.title(title, **fontstyle)
    plt.ylim(y_limits)

    plt.legend(framealpha=0.8, frameon=True, labelspacing=0.01, loc='lower right', fontsize=18)

    if not no_display:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trained agent")
    parser.add_argument('-i', '--input-dir', help='folder with the plots as npz files', type=str, required=True)
    parser.add_argument('-t', '--title', help='Plot title', type=str, default='Learning Curve')
    parser.add_argument('--episode_window', type=int, default=40,
                        help='Episode window for moving average plot (default: 40)')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Change the y_limit to correspond shaped reward bounds')
    parser.add_argument('--y-lim', nargs=2, type=float, default=[-1, -1], help="limits for the y axis")
    parser.add_argument('--truncate-x', type=int, default=-1,
                        help="Truncate the experiments after n ticks on the x-axis (default: -1, no truncation)")
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

    plots = [f for f in os.listdir(args.input_dir) if f.endswith('.npz')]
    plots.sort()

    if len(plots) == 0:
        printRed("No npz files found in {}".format(args.input_dir))
        exit(-1)

    comparePlots(args.input_dir, plots, title=args.title, y_limits=y_limits, no_display=args.no_display,
                timesteps=args.timesteps, truncate_x=args.truncate_x)
