"""
Create csv result from a folder of methods
"""
from __future__ import print_function, division, absolute_import

import argparse
import json
import os

import pandas as pd
import numpy as np

from replay.aggregate_plots import loadEpisodesData


parser = argparse.ArgumentParser(description='Create a report file for a given algo')
parser.add_argument('-i', '--log-dir', type=str, default="", required=True, help='Path to a base log folder (environment level)')
args = parser.parse_args()

assert os.path.isdir(args.log_dir), "--log-dir must be a path to a valid folder"

log_dir = args.log_dir

# Add here keys from exp_config.json that should be saved in the csv report file
exp_configs = {'srl_model_path': []}

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
            try:
                env_globals = json.load(open(path + "env_globals.json", 'r'))
                train_args = json.load(open(path + "args.json", 'r'))
            except FileNotFoundError:
                print("config files not found for {}".format(exp))
                continue

            for key in exp_configs.keys():
                exp_configs[key].append(env_globals.get(key, None))
            break



exp_configs.update({'methods': methods, 'rl_algo': algos})

# Export to csv
result_df = pd.DataFrame(exp_configs)
result_df.to_csv('{}/results.csv'.format(log_dir), sep=",", index=False)
print("Saved results to {}/results.csv".format(log_dir))
