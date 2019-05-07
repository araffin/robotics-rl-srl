"""
Plot past experiment in visdom
"""
import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc

from replay.aggregate_plots import lightcolors, darkcolors, Y_LIM_SHAPED_REWARD, Y_LIM_SPARSE_REWARD, millions
from rl_baselines.cross_eval import dict2array
from rl_baselines.visualize import smoothRewardCurve
# Init seaborn
sns.set()
# Style for the title
fontstyle = {'fontname': 'DejaVu Sans', 'fontsize': 20, 'fontweight': 'bold'}
rc('font', weight='bold')


def crossEvalPlot(res, tasks, title, y_limits):
    y_array = res[:, :, 1:]
    #    y_array = np.sort(res[:, :, 1:], axis=2)
    #     y_array = np.mean(y_array[:,:,1:],axis=2)

    x = res[:, :, 0][0]

    fig = plt.figure(title)
    for i in range(len(y_array)):
        label = tasks[i]
        y = y_array[i].T
        print('{}: {} experiments'.format(label, len(y)))
        # Compute mean for different seeds
        m = np.mean(y, axis=0)
        # Compute standard error
        s = np.squeeze(np.asarray(np.std(y, axis=0)))
        n = y.shape[0]
        plt.fill_between(x, m - s / np.sqrt(n), m + s / np.sqrt(n), color=lightcolors[i % len(lightcolors)], alpha=0.5)
        plt.plot(x, m, color=darkcolors[i % len(darkcolors)], label=label, linewidth=2)

    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards', fontsize=20, fontweight='bold')

    plt.title(title, **fontstyle)
    if (y_limits[0] != y_limits[1]):
        plt.ylim(y_limits)

    plt.legend(framealpha=0.8, frameon=True, labelspacing=0.01, loc='lower right', fontsize=18)
    plt.show()


def smoothPlot(res, tasks, title, y_limits):
    y = np.mean(res[:, :, 1:], axis=2)
    #    y = np.sort(res[:, :, 1:], axis=2)
    #    y = np.mean(y[:,:,1:],axis=2)
    x = res[:, :, 0][0]
    print(y.shape, x.shape)
    fig = plt.figure(title)
    for i in range(len(y)):
        label = tasks[i]
        tmp_x, tmp_y = smoothRewardCurve(x, y[i], conv_len=3)
        plt.plot(tmp_x, tmp_y, color=darkcolors[i % len(darkcolors)], label=label, linewidth=2)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards', fontsize=20, fontweight='bold')

    plt.title(title, **fontstyle)
    if (y_limits[0] != y_limits[1]):
        plt.ylim(y_limits)

    plt.legend(framealpha=0.8, frameon=True, labelspacing=0.01, loc='lower right', fontsize=18)
    plt.show()



#Example command:
# python -m replay.cross_eval_plot -i logs/sc2cc/OmnirobotEnv-v0/srl_combination/ppo2/19-04-29_14h59_35/episode_eval.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot the learning curve during a training for different tasks")
    parser.add_argument('-i', '--input-path', help='folder with the plots as pkl files', type=str, required=True)
    parser.add_argument('-t', '--title', help='Plot title', type=str, default='Learning Curve')
    # parser.add_argument('--episode_window', type=int, default=40,
    #                     help='Episode window for moving average plot (default: 40)')
    # parser.add_argument('--shape-reward', action='store_true', default=False,
    #                     help='Change the y_limit to correspond shaped reward bounds')
    parser.add_argument('--y-lim', nargs=2, type=float, default=[-1, -1], help="limits for the y axis")
    parser.add_argument('--truncate-x', type=int, default=-1,
                        help="Truncate the experiments after n ticks on the x-axis (default: -1, no truncation)")
    # parser.add_argument('--timesteps',
    #                     help='Plot timesteps instead of episodes')
    parser.add_argument('--eval-tasks', type=str, nargs='+', default=['Circular', 'Target Reaching','Square'],
                        help='A cross evaluation from the latest stored model to all tasks')
    parser.add_argument('-s','--smooth', action='store_true', default=False,
                        help='Plot with a smooth mode')
    args = parser.parse_args()


    load_path = args.input_path
    title     = args.title
    y_limits  = args.y_lim
    tasks     = args.eval_tasks



    assert (os.path.isfile(load_path) and load_path.split('.')[-1]=='pkl'), 'Please load a valid .pkl file'


    with open(load_path, "rb") as file:
        data = pickle.load(file)

    res = dict2array(['cc', 'sc'], data)

    if(args.smooth):
        smoothPlot(res,tasks,title,y_limits)
    else:
        crossEvalPlot(res, tasks, title,y_limits)
