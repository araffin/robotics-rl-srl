"""
Create csv result from a folder of methods
"""
from __future__ import print_function, division, absolute_import

import argparse
import json
import os
import glob

import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(description='Create a report file for a given algo')
parser.add_argument('-i', '--log-dir', type=str, default="", required=True, help='Path to a base log folder (environment level)')
parser.add_argument('--timestep-budget', type=int, default=None, help='the timesteps budget')
parser.add_argument('--min-timestep', type=int, default=None, help='the minimum timesteps for a monitoring to count')
parser.add_argument('--episode-window', type=int, default=100, help='The expected reward over the number of episodes')
args = parser.parse_args()

assert os.path.isdir(args.log_dir), "--log-dir must be a path to a valid folder"

log_dir = args.log_dir

# Add here keys from exp_config.json that should be saved in the csv report file
exp_configs = {'srl_model_path': []}
exp_results = {'mean_reward': [],
               'std_reward': []}

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
        for exp in os.listdir(exp_dir):
            path = "{}/{}/".format(exp_dir, exp)
            print(path)
            data = []
            env_globals = None
            for sess_id, session in enumerate(glob.glob(path + "/*")):
                try:
                    env_globals = json.load(open(session + "/env_globals.json", 'r'))
                    train_args = json.load(open(session + "/args.json", 'r'))
                    pass
                except FileNotFoundError:
                    print("config files not found for {}".format(exp))
                    continue

                run_acc = None
                monitor_files = sorted(glob.glob(session + "/*.monitor.csv"))
                for monitor_file in monitor_files:
                    run = np.array(pd.read_csv(monitor_file, skiprows=1)[["l", "r"]])
                    if run_acc is None:
                        run_acc = run
                    else:
                        # helps with uneven runs
                        if run.shape[0] < run_acc.shape[0]:
                            run_acc = run_acc[:run.shape[0], :]
                        run_acc += run[:run_acc.shape[0], :]

                if run_acc is not None and (args.min_timestep is None or np.sum(run_acc[:, 0]) > args.min_timestep):
                    run_acc[:, 1] = run_acc[:, 1] / len(monitor_files)
                    run_acc[:, 0] = np.cumsum(run_acc[:, 0])
                    if args.timestep_budget is not None:
                        run_acc = run_acc[run_acc[:, 0] < args.timestep_budget]

                    data.append(run_acc[-args.episode_window:, 1])

            if len(data) > 0 and env_globals is not None:
                mean_rew = np.mean(data)
                std_rew = np.std(data)
                exp_results['mean_reward'].append(mean_rew)
                exp_results['std_reward'].append(std_rew)
                for key in exp_configs.keys():
                    exp_configs[key].append(env_globals.get(key, None))


exp_configs.update({'methods': methods, 'rl_algo': algos, **exp_results})
# Export to csv
result_df = pd.DataFrame(exp_configs)
result_df.to_csv('{}/results.csv'.format(log_dir), sep=",", index=False)
print("Saved results to {}/results.csv".format(log_dir))
