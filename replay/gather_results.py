"""
Create csv result from a folder of methods
"""
from __future__ import print_function, division, absolute_import

import argparse
import json
import os
import glob
from collections import OrderedDict

import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description='Create a report file for a given algo')
parser.add_argument('-i', '--log-dir', type=str, default="", required=True, help='Path to a base log folder (environment level)')
parser.add_argument('--timestep-budget', type=int, nargs='+',  default=[], help='the timesteps budget')
parser.add_argument('--min-timestep', type=int, default=None, help='the minimum timesteps for a monitoring to count')
parser.add_argument('--episode-window', type=int, default=100, help='The expected reward over the number of episodes')
args = parser.parse_args()

assert os.path.isdir(args.log_dir), "--log-dir must be a path to a valid folder"

log_dir = args.log_dir

args.timestep_budget = sorted(list(set(args.timestep_budget)))

# Add here keys from exp_config.json that should be saved in the csv report file
exp_configs = [('srl_model_path', [])]
if len(args.timestep_budget) > 0:
    exp_results = []
    for ts_budget in args.timestep_budget:
        exp_results.append(('mean_reward_{}'.format(ts_budget), []))
        exp_results.append(('std_reward_{}'.format(ts_budget), []))
else:
    exp_results = [('mean_reward', []),
                   ('std_reward', [])]

exp_configs = OrderedDict(exp_configs)
exp_results = OrderedDict(exp_results)

methods = []
algos = []

for method in os.listdir(log_dir):
    algos_dir = "{}/{}/".format(log_dir, method)
    if not os.path.isdir(algos_dir):
        continue
    for algo in os.listdir(algos_dir):
        if not os.path.isdir(algos_dir + algo):
            continue
        algos.append(algo)
        methods.append(method)
        exp_dir = "{}/{}/".format(algos_dir, algo)
        print(exp_dir)
        data = [[] for _ in args.timestep_budget]  # the RL data for all the requested budgets
        env_globals = None
        for exp_id, exp in enumerate(os.listdir(exp_dir)):
            path = "{}/{}/".format(exp_dir, exp)

            try:
                env_globals = json.load(open(path + "/env_globals.json", 'r'))
                train_args = json.load(open(path + "/args.json", 'r'))
                pass
            except FileNotFoundError:
                print("config files not found for {}".format(exp))
                continue

            run_acc = None  # the accumulated RL data for the run
            monitor_files = sorted(glob.glob(path + "/*.monitor.csv"))
            for monitor_file in monitor_files:
                run = np.array(pd.read_csv(monitor_file, skiprows=1)[["l", "r"]])
                if run_acc is None:
                    run_acc = run
                else:
                    # helps with uneven runs
                    if run.shape[0] < run_acc.shape[0]:
                        run_acc = run_acc[:run.shape[0], :]
                    run_acc += run[:run_acc.shape[0], :]

            # make sure there is data here, and that there it is above the minimum timestep threashold
            if run_acc is not None and (args.min_timestep is None or np.sum(run_acc[:, 0]) > args.min_timestep):
                run_acc[:, 1] = run_acc[:, 1] / len(monitor_files)
                run_acc[:, 0] = np.cumsum(run_acc[:, 0])
                if len(args.timestep_budget) > 0:  # extract the episodes for the requested budget
                    for i, ts_budget in enumerate(args.timestep_budget):
                        if np.all(run_acc[:, 0] < ts_budget):
                            print("warning, budget too high for {} using {}, the highest logged wil be for {} timesteps."
                                  .format(algo, method, np.max(run_acc[:, 0])))
                        budget_acc = run_acc[run_acc[:, 0] < ts_budget]
                        if budget_acc.shape[0] == 0:
                            print("budget too low for {} using {}".format(algo, method))
                            continue
                        data[i].append(budget_acc[-args.episode_window:, 1])
                else:  # otherwise do for the highest value possible
                    data.append(run_acc[-args.episode_window:, 1])

        if len(data) > 0 and env_globals is not None:
            if len(args.timestep_budget) > 0:  # mean and std for every budget requested
                for i, ts_budget in enumerate(args.timestep_budget):
                    mean_rew = np.mean(data[i])
                    std_rew = np.std(data[i])
                    exp_results['mean_reward_{}'.format(ts_budget)].append(mean_rew)
                    exp_results['std_reward_{}'.format(ts_budget)].append(std_rew)
            else:
                mean_rew = np.mean(data)
                std_rew = np.std(data)
                exp_results['mean_reward'].append(mean_rew)
                exp_results['std_reward'].append(std_rew)
            for key in exp_configs.keys():
                exp_configs[key].append(env_globals.get(key, None))


exp_configs.update({'methods': methods, 'rl_algo': algos})
exp_configs.update(exp_results)
# Export to csv
result_df = pd.DataFrame(exp_configs)
result_df.to_csv('{}/results.csv'.format(log_dir), sep=",", index=False)
print("Saved results to {}/results.csv".format(log_dir))
