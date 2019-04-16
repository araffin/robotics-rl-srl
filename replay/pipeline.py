import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import json

from rl_baselines.visualize import  movingAverage, loadCsv,loadData
from replay.aggregate_plots import lightcolors, darkcolors, Y_LIM_SHAPED_REWARD, Y_LIM_SPARSE_REWARD, millions
from srl_zoo.utils import printGreen, printRed, printYellow

# Init seaborn
sns.set()
# Style for the title
fontstyle = {'fontname': 'DejaVu Sans', 'fontsize': 16}




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


def plotGatheredData(x_list,y_list,y_limits, timesteps,title,legends,no_display,truncate_x=-1,normalization=False):
    assert len(legends)==len(y_list)
    printGreen("{} Experiments".format(len(y_list)))
    #print("Min, Max rewards:", np.min(y_list), np.max(y_list))

    lengths = list(map(len, x_list))
    min_x, max_x = np.min(lengths), np.max(lengths)


    if truncate_x > 0:
        min_x = min(truncate_x, min_x)
    x = np.array(x_list[0][:min_x])

    print(lengths,x.shape)
    fig = plt.figure(title)
    for i in range(len(y_list)):
        label = legends[i]
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
    plt.ylim(y_limits)

    plt.legend(framealpha=0.8, frameon=True, labelspacing=0.01, loc='lower right', fontsize=16)

    if not no_display:
        plt.show()




def GatherExperiments(folders, algo,  window=40, title="", min_num_x=-1,
                            timesteps=False, output_file="",):
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
    # if output_file != "":
    #     printGreen("Saving aggregated data to {}.npz".format(output_file))
    #     np.savez(output_file, x=x, y=y)
    return x,y


def comparePlots(path,  algo,y_limits,title="Learning Curve",
                 timesteps=False, truncate_x=-1, no_display=False,normalization=False):
    """
    :param path: (str) path to the folder where the plots are stored
    :param plots: ([str]) List of saved plots as npz file
    :param y_limits: ([float]) y-limits for the plot
    :param title: (str) plot title
    :param timesteps: (bool) Plot timesteps instead of episodes
    :param truncate_x: (int) Truncate the experiments after n ticks on the x-axis
    :param no_display: (bool) Set to true, the plot won't be displayed (useful when only saving plot)
    """

    folders = []
    other = []
    legends=[]
    for folder in os.listdir(path):
        folders_srl=[]
        other_srl=[]
        tmp_path = "{}/{}/{}/".format(path, folder, algo)
        legends.append(folder)
        for f in os.listdir(tmp_path):
            paths = "{}/{}/{}/{}/".format(path, folder, algo,f)
            env_globals = json.load(open(paths + "env_globals.json", 'r'))
            train_args = json.load(open(paths + "args.json", 'r'))
            if train_args["shape_reward"] == args.shape_reward:
                folders_srl.append(paths)
            else:
                other_srl.append(paths)
        folders.append(folders_srl)
        other.append(other_srl)


    x_list,y_list=[],[]
    for folders_srl in folders:
        x,y=GatherExperiments(folders_srl, algo,  window=40, title=title, min_num_x=-1,
                          timesteps=timesteps, output_file="")
        x_list.append(x)
        y_list.append(y)

    plotGatheredData(x_list,y_list,y_limits,timesteps,title,legends,no_display,truncate_x,normalization)



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
    parser.add_argument('--algo',type=str,default='ppo2',help='The RL algorithms result to show')
    parser.add_argument('--norm', action='store_true', default=False, help='To normalize the output by the maximum reward')
    #
    # parser.add_argument('--tasks', type=str, nargs='+', default=["cc"],
    #                     help='The tasks for the robot',
    #                     choices=["cc", "ec", "sqc", "sc"])



    args = parser.parse_args()

    y_limits = args.y_lim
    if y_limits[0] == y_limits[1]:
        if args.shape_reward:
            y_limits = Y_LIM_SHAPED_REWARD
        else:
            y_limits = Y_LIM_SPARSE_REWARD
        print("Using default limits:", y_limits)


    ALGO_NAME=args.algo


    x_list=[]
    y_list=[]

    comparePlots(args.input_dir, args.algo, title=args.title, y_limits=y_limits, no_display=args.no_display,
            timesteps=args.timesteps, truncate_x=args.truncate_x,normalization=args.norm)


#python -m replay.pipeline -i logs/OmnirobotEnv-v0/ --algo ppo2 --title cc