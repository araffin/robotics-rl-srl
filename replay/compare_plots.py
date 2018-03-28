import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from replay.aggregate_plots import lightcolors, darkcolors, Y_LIM_SHAPED_REWARD, Y_LIM_SPARSE_REWARD, millions
from srl_priors.utils import printGreen

# Init seaborn
sns.set()
# Style for the title
fontstyle = {'fontname': 'DejaVu Sans', 'fontsize': 16}


def comparePlots(path, plots, title="Learning Curve",
                 shaped_reward=False, timesteps=False):
    """
    :param path: (str) path to the folder where the plots are stored
    :param plots: ([str]) List of saved plots as npz file
    :param title: (str) plot title
    :param timesteps: (bool) Plot timesteps instead of episodes
    :param shaped_reward: (bool)
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
        plt.xlabel('Number of Timesteps')
        fig.axes[0].xaxis.set_major_formatter(formatter)
    else:
        plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards')

    plt.title(title, **fontstyle)
    if shaped_reward:
        plt.ylim(Y_LIM_SHAPED_REWARD)
    else:
        plt.ylim(Y_LIM_SPARSE_REWARD)

    plt.legend(framealpha=0.5, labelspacing=0.01, loc='lower right', fontsize=16)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot trained agent")
    parser.add_argument('-i', '--input-dir', help='folder with the plots as npz files', type=str, required=True)
    parser.add_argument('--episode_window', type=int, default=40,
                        help='Episode window for moving average plot (default: 40)')
    parser.add_argument('--shape-reward', action='store_true', default=False,
                        help='Change the y_limit to correspond shaped reward bounds')
    parser.add_argument('--timesteps', action='store_true', default=False,
                        help='Plot timesteps instead of episodes')
    args = parser.parse_args()

    plots = [f for f in os.listdir(args.input_dir) if f.endswith('.npz')]

    comparePlots(args.input_dir, plots, timesteps=args.timesteps, shaped_reward=args.shape_reward)
