"""
Plot past experiment in visdom
"""
import argparse
import json

from visdom import Visdom

from rl_baselines.visualize import timestepsPlot, episodePlot

parser = argparse.ArgumentParser(description="Plot trained agent using Visdom")
parser.add_argument('--log-dir', help='folder with the saved agent model', required=True)
parser.add_argument('--episode_window', type=int, default=40,
                    help='Episode window for moving average plot (default: 40)')
args = parser.parse_args()

viz = Visdom()

env_globals = json.load(open(args.log_dir + "kuka_env_globals.json", 'r'))
train_args = json.load(open(args.log_dir + "args.json", 'r'))

srl_model = train_args['srl_model'] if train_args['srl_model'] != "" else "raw pixels"
episodePlot(viz, None, args.log_dir, train_args['env'], train_args['algo'],
            title=srl_model + " [Episodes]", window=args.episode_window)
timestepsPlot(viz, None, args.log_dir, train_args['env'], train_args['algo'], title=srl_model)
