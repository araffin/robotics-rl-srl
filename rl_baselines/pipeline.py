"""
baseline benchmark script for openAI RL Baselines
"""
import os
import argparse
import subprocess

import tensorflow as tf
import numpy as np

from srl_priors.utils import printGreen, printRed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow

def main():
    parser = argparse.ArgumentParser(description="OpenAI RL Baselines Benchmark", 
        epilog='After the arguments are parsed, the rest are assumed to be arguments for rl_baselines.train')
    parser.add_argument('--algo', type=str, default='ppo2', help='OpenAI baseline to use',
                        choices=['acer', 'deepq', 'a2c', 'ppo2', 'random_agent', 'ddpg', 'cma-es', 'ars'])
    parser.add_argument('--env', type=str, nargs='+', default=["KukaButtonGymEnv-v0"], help='environment ID(s)',
                        choices=["KukaButtonGymEnv-v0", "KukaRandButtonGymEnv-v0", 
                        "Kuka2ButtonGymEnv-v0", "KukaMovingButtonGymEnv-v0"])
    parser.add_argument('--srl-model', type=str, nargs='+', default=["raw_pixels"], help='SRL model(s) to use',
                        choices=["autoencoder", "ground_truth", "srl_priors", "supervised", 
                        "pca", "vae", "joints", "joints_position", "raw_pixels"])
    parser.add_argument('--num-timesteps', type=int, default=1e6, help='number of timesteps the baseline should run')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display baseline STDOUT')
    parser.add_argument('--num-iteration', type=int, default=15, help='number of time each algorithm should be ' + \
                        'run for each unique combination of environment and srl-model.')
    parser.add_argument('--seed', type=int, default=0, 
                        help='initial seed for each unique combination of environment and srl-model.')

    # returns the parsed arguments, and the rest are assumed to be arguments for rl_baselines.train
    args, train_args = parser.parse_known_args()

    # Sanity check
    assert args.num_timesteps >= 1, "Error: --num-timesteps cannot be less than 1"

    # Removing duplicates
    args.srl_model = list(set(args.srl_model))
    args.env = list(set(args.env))

    # the seeds used in training the baseline.
    seeds = list(np.arange(args.num_iteration) + args.seed)

    if args.verbose:
        # None here means stdout of terminal for subprocess.call
        stdout = None
    else:
        stdout = open(os.devnull, 'w')

    printGreen("\nRunning {} benchmarks {} times...".format(args.algo, args.num_iteration))
    print("\nSRL-Models:\t{}".format(models))
    print("environments:\t{}".format(envs))
    print("verbose:\t{}".format(args.verbose))
    print("timesteps:\t{}".format(args.num_timesteps))
    for model in models:
        for env in envs:
            for i in range(args.num_iteration):

                printGreen("\nIteration_num={} (seed: {}), Environment='{}', SRL-Model='{}'".format(i, seeds[i], env, model))

                # redefine the parsed args for rl_baselines.train
                if model != "raw_pixels":
                    # raw_pixels is when --srl-model is left as default
                    train_args.extend(['--srl-model', model])
                train_args.extend(['--seed', str(seeds[i]), '--algo', args.algo, '--env', env, '--num-timesteps',
                                   str(int(args.num_timesteps))])

                ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + train_args, stdout=stdout)

                if ok != 0:
                    printRed("An error occured, error code: {}".format(ok))
                    # throw the error down to the terminal
                    return ok


if __name__ == '__main__':
    main()
