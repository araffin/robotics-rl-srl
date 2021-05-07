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
from matplotlib.ticker import FuncFormatter
from replay.aggregate_plots import lightcolors, darkcolors, Y_LIM_SHAPED_REWARD, Y_LIM_SPARSE_REWARD, millions
from rl_baselines.cross_eval import dict2array
from rl_baselines.visualize import smoothRewardCurve
# Init seaborn
sns.set()
# Style for the title
fontstyle = {'fontname': 'DejaVu Sans', 'fontsize': 24, 'fontweight': 'bold'}
rc('font', weight='bold')


def crossEvalPlot(res, tasks, title, y_limits , truncate_x, plot_mean=False, timesteps=False):

    index_x = -1
    episodes = res[:,:,0][0]
    if (truncate_x>-1):
        for eps in episodes:
            index_x += 1
            if(eps >= truncate_x):
                break
    if(index_x ==-1 ):
        y_array = res[:, :, 1:]
        x = res[:, :, 0][0]
    else:
        y_array = res[:, :index_x, 1:]
        x = res[:, :index_x, 0][0]
    if(timesteps):
        x = 250 * x
    sum_mean = []
    sum_s    = []
    sum_n    = []

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
        sum_mean +=[m]
        sum_s    +=[s]
        sum_n    +=[n]
        plt.fill_between(x, m - s / np.sqrt(n), m + s / np.sqrt(n), color=lightcolors[i % len(lightcolors)], alpha=0.5)
        plt.plot(x, m, color=darkcolors[i % len(darkcolors)], label=label, linewidth=2)

    #reward_sum = np.concatenate([res[0, :index_x, 1:], res[1, :index_x, 1:]], axis=1)

    if(plot_mean):

        m = np.mean(sum_mean, axis=0)
        # Compute standard error
        s = np.mean(sum_s)
        n = np.mean(n)
        plt.fill_between(x, m - s / np.sqrt(n), m + s / np.sqrt(n), color=lightcolors[4 % len(lightcolors)], alpha=0.5)
        plt.plot(x, m, color=darkcolors[4 % len(darkcolors)], label='mean reward', linewidth=2)

    if timesteps:
        formatter = FuncFormatter(millions)
        plt.xlabel('Number of Timesteps')
        fig.axes[0].xaxis.set_major_formatter(formatter)
    else:
        plt.xlabel('Number of Episodes')
    plt.ylabel('Normalized Rewards', fontsize=15 , fontweight='bold')

    plt.title(title, **fontstyle)
    if (y_limits[0] != y_limits[1]):
        plt.ylim(y_limits)

    plt.legend(framealpha=0.8, frameon=True, labelspacing=0.01, loc='lower right', fontsize=20)
    plt.show()


def smoothPlot(res, tasks, title, y_limits,timesteps):
    y = np.mean(res[:, :, 1:], axis=2)
    #    y = np.sort(res[:, :, 1:], axis=2)
    #    y = np.mean(y[:,:,1:],axis=2)
    x = res[:, :, 0][0]
    if(timesteps):
        x = x*250
    print(y.shape, x.shape)
    fig = plt.figure(title)
    for i in range(len(y)):
        label = tasks[i]
        tmp_x, tmp_y = smoothRewardCurve(x, y[i], conv_len=2)
        plt.plot(tmp_x, tmp_y, color=darkcolors[i % len(darkcolors)], label=label, linewidth=2)
    if (timesteps):
        formatter = FuncFormatter(millions)
        plt.xlabel('Number of Timesteps')
        fig.axes[0].xaxis.set_major_formatter(formatter)
    else:
        plt.xlabel('Number of Episodes')
    plt.ylabel('Rewards', fontsize=20, fontweight='bold')

    plt.title(title, **fontstyle)
    if (y_limits[0] != y_limits[1]):
        plt.ylim(y_limits)

    plt.legend(framealpha=0.8, frameon=True, labelspacing=0.01, loc='lower right', fontsize=18)
    plt.show()



#Example command:
# python -m replay.cross_eval_plot -i logs/sc2cc/OmnirobotEnv-v0/srl_combination/ppo2/19-04-29_14h59_35/eval.pkl
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot the learning curve during a training for different tasks")
    parser.add_argument('-i', '--input-path', help='folder with the plots as pkl files', type=str, required=True)
    parser.add_argument('-t', '--title', help='Plot title', type=str, default='Learning Curve')

    parser.add_argument('--y-lim', nargs=2, type=float, default=[-1, -1], help="limits for the y axis")
    parser.add_argument('--truncate-x', type=int, default=-1,
                        help="Truncate the experiments after n ticks on the x-axis (default: -1, no truncation)")
    parser.add_argument('--eval-tasks', type=str, nargs='+', default=['cc', 'esc','sc'],
                        help='A cross evaluation from the latest stored model to all tasks')
    parser.add_argument('-s','--smooth', action='store_true', default=False,
                        help='Plot with a smooth mode')
    parser.add_argument('--timesteps', action = 'store_true',default=False,
                        help='Plot with timesteps')

    args = parser.parse_args()


    load_path = args.input_path
    title     = args.title
    y_limits  = args.y_lim
    tasks     = args.eval_tasks
    truncate_x= args.truncate_x


    assert (os.path.isfile(load_path) and load_path.split('.')[-1]=='pkl'), 'Please load a valid .pkl file'


    with open(load_path, "rb") as file:
        data = pickle.load(file)


    res = dict2array(args.eval_tasks, data)

    print("{} episodes evaluations to plot".format(res.shape[1]))
    if(args.smooth):
        smoothPlot(res[:,1:],tasks,title,y_limits,args.timesteps)
    else:
        crossEvalPlot(res[:,:], tasks, title,y_limits, truncate_x,args.timesteps)
