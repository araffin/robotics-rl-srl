"""
Modified version of https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/visualize.py
Script used to send plot data to visdom
"""
import glob
import os

import numpy as np
from scipy.signal import medfilt


def smoothRewardCurve(x, y, conv_len=30):
    """
    :param x: (numpy array)
    :param y: (numpy array)
    :param conv_len: an integer, kernel size of the convolution
    :return: (numpy array, numpy array)
    """
    # Halfwidth of our smoothing convolution
    halfwidth = min(conv_len+1, int(np.ceil(len(x) / conv_len)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
            np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fixPoint(x, y, interval):
    """
    :param x: (numpy array)
    :param y: (numpy array)
    :param interval: (int)
    :return: (numpy array, numpy array)
    """
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                    (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def loadCsv(log_folder, is_es=False):
    """
    Load data (rewards for each episode) from csv file (generated by gym monitor)
    :param log_folder: (str)
    :param is_es: (bool) Set the loading to get the best agent from the envs
    :return: (numpy array, numpy array)
    """
    datas = []
    monitor_files = glob.glob(os.path.join(log_folder, '*.monitor.csv'))

    for input_file in monitor_files:
        data = []
        with open(input_file, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                tmp = [t_time, int(tmp[1]), float(tmp[0])]
                data.append(tmp)
        datas.append(data)

    if is_es:
        max_data_len = np.max([len(data) for data in datas])

        # get the reward for each file, and make sure they are all the same length
        r_list = []
        for data in datas:
            if len(data) == max_data_len:
                r_list.append([x[2] for x in data])
            else:
                r_list.append([x[2] for x in data] + ((max_data_len - len(data)) * [-np.inf]))
        max_sess = np.argmax(r_list, axis=0)

        result = []
        timesteps = 0
        for i in range(max_data_len):
            result.append([timesteps, datas[max_sess[i]][i][-1]])
            timesteps += datas[max_sess[i]][i][1]
    else:
        datas = [x for data in datas for x in data]
        datas = sorted(datas, key=lambda d_entry: d_entry[0])
        result = []
        timesteps = 0
        for i in range(len(datas)):
            result.append([timesteps, datas[i][-1]])
            timesteps += datas[i][1]

    return result, timesteps


def loadData(log_folder, smooth, bin_size, is_es=False):
    """
    :param log_folder: (str)
    :param smooth: (int) Smoothing method
    :param bin_size: (int)
    :param is_es: (bool)
    :return:
    """
    result, timesteps = loadCsv(log_folder, is_es=is_es)

    if len(result) < bin_size:
        return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smoothRewardCurve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    x, y = fixPoint(x, y, bin_size)
    return [x, y]


def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def episodePlot(viz, win, folder, game, name, window=5, title="", is_es=False):
    """
    Create/Update a vizdom plot of reward per episode
    :param viz: (visdom object)
    :param win: (str) Window name, it is the unique id of each plot
    :param folder: (str) Log folder, where the monitor.csv is stored
    :param game: (str) Name of the environment
    :param name: (str) Algo name
    :param window: (int) Smoothing window
    :param title: (str) additional info to display in the plot title
    :param is_es: (bool)
    :return: (str)
    """
    result, _ = loadCsv(folder, is_es=is_es)

    if len(result) == 0:
        return win

    y = np.array(result)[:, 1]
    x = np.arange(len(y))

    if y.shape[0] < window:
        return win

    y = movingAverage(y, window)

    if len(y) == 0:
        return win

    # Truncate x
    x = x[len(x) - len(y):]
    opts = {
        "title": "{}\n{}".format(game, title),
        "xlabel": "Number of Episodes",
        "ylabel": "Rewards",
        "legend": [name]
    }
    return viz.line(y, x, win=win, opts=opts)


def timestepsPlot(viz, win, folder, game, name, bin_size=100, smooth=1, title="", is_es=False):
    """
    Create/Update a vizdom plot of reward per timesteps
    :param viz: (visdom object)
    :param win: (str) Window name, it is the unique id of each plot
    :param folder: (str) Log folder, where the monitor.csv is stored
    :param game: (str) Name of the environment
    :param name: (str) Algo name
    :param bin_size: (int)
    :param smooth: (int) Smoothing method (0 for no smoothing)
    :param title: (str) additional info to display in the plot title
    :param is_es: (bool)
    :return: (str)
    """
    tx, ty = loadData(folder, smooth, bin_size, is_es=is_es)
    if tx is None or ty is None:
        return win

    if len(tx) * len(ty) == 0:
        return win

    tx, ty = np.array(tx), np.array(ty)

    opts = {
        "title": "{}\n{}".format(game, title),
        "xlabel": "Number of Timesteps",
        "ylabel": "Rewards",
        "legend": [name]
    }
    return viz.line(ty, tx, win=win, opts=opts)

def episodesEvalPlot(viz, win, folder, game, name, window=1, title=""):

    folder+='episode_eval.npy'
    if(os.path.isfile(folder)):
        result = np.load(folder)
    else:
        return win

    if len(result) == 0:
        return win
    print(result.shape)
   
    y = np.mean(result[:, :, 1:], axis=2).T


    x = result[:, :, 0].T

    if y.shape[0] < window:
        return win

    if len(y) == 0:
        return win


    opts = {
        "title": "{}\n{}".format(game, title),
        "xlabel": "Episodes",
        "ylabel": "Rewards",
        "legend": name
    }

    return viz.line(y, x, win=win, opts=opts)