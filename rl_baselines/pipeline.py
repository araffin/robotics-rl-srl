"""
baseline benchmark script for openAI RL Baselines
"""
import os
import argparse
import subprocess

import yaml
import numpy as np

from rl_baselines.registry import registered_rl
from environments.registry import registered_env
from state_representation.registry import registered_srl
from state_representation import SRLType
from srl_zoo.utils import printGreen, printRed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow


def main():
    parser = argparse.ArgumentParser(description="OpenAI RL Baselines Benchmark",
                                     epilog='After the arguments are parsed, the rest are assumed to be arguments for' +
                                            ' rl_baselines.train')
    parser.add_argument('--algo', type=str, default='ppo2', help='OpenAI baseline to use',
                        choices=list(registered_rl.keys()))
    parser.add_argument('--env', type=str, nargs='+', default=["KukaButtonGymEnv-v0"], help='environment ID(s)',
                        choices=list(registered_env.keys()))
    parser.add_argument('--srl-model', type=str, nargs='+', default=["raw_pixels"], help='SRL model(s) to use',
                        choices=list(registered_srl.keys()))
    parser.add_argument('--num-timesteps', type=int, default=1e6, help='number of timesteps the baseline should run')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display baseline STDOUT')
    parser.add_argument('--num-iteration', type=int, default=15,
                        help='number of time each algorithm should be run for each unique combination of environment ' +
                             ' and srl-model.')
    parser.add_argument('--seed', type=int, default=0,
                        help='initial seed for each unique combination of environment and srl-model.')

    # returns the parsed arguments, and the rest are assumed to be arguments for rl_baselines.train
    args, train_args = parser.parse_known_args()

    # Sanity check
    assert args.num_timesteps >= 1, "Error: --num-timesteps cannot be less than 1"
    assert args.num_iteration >= 1, "Error: --num-iteration cannot be less than 1"

    # Removing duplicates and sort
    srl_models = list(set(args.srl_model))
    envs = list(set(args.env))
    srl_models.sort()
    envs.sort()

    # loading the config file for the srl_models
    with open('config/srl_models.yaml', 'rb') as f:
        all_models = yaml.load(f)

    # Checking definition and presence of all requested srl_models
    valid = True
    for env in envs:
        # validated the env definition
        if env not in all_models:
            printRed("Error: 'srl_models.yaml' missing definition for environment {}".format(env))
            valid = False
            continue  # skip to the next env, this one is not valid

        # checking log_folder for current env
        missing_log = "log_folder" not in all_models[env]
        if missing_log:
            printRed("Error: 'srl_models.yaml' missing definition for log_folder in environment {}".format(env))
            valid = False

        # validate each model for the current env definition
        for model in srl_models:
            if registered_srl[model][0] == SRLType.ENVIRONMENT:
                continue  # not an srl model, skip to the next model
            elif model not in all_models[env]:
                printRed("Error: 'srl_models.yaml' missing srl_model {} for environment {}".format(model, env))
                valid = False
            elif (not missing_log) and (not os.path.exists(all_models[env]["log_folder"] + all_models[env][model])):
                # checking presence of srl_model path, if and only if log_folder exists
                printRed("Error: srl_model {} for environment {} was defined in ".format(model, env) +
                         "'srl_models.yaml', however the file {} it was tagetting does not exist.".format(
                             all_models[env]["log_folder"] + all_models[env][model]))
                valid = False

    assert valid, "Errors occured due to malformed 'srl_models.yaml', cannot continue."

    # check that all the SRL_models can be run on all the environments
    valid = True
    for env in envs:
        for model in srl_models:
            if registered_srl[model][1] is not None:
                found = False
                for compatible_class in registered_srl[model][1]:
                    if issubclass(compatible_class, registered_env[env][0]):
                        found = True
                        break
                if not found:
                    valid = False
                    printRed("Error: srl_model {}, is not compatible with the {} environment.".format(model, env))
    assert valid, "Errors occured due to an incompatible combination of srl_model and environment, cannot continue."

    # the seeds used in training the baseline.
    seeds = list(np.arange(args.num_iteration) + args.seed)

    if args.verbose:
        # None here means stdout of terminal for subprocess.call
        stdout = None
    else:
        stdout = open(os.devnull, 'w')

    printGreen("\nRunning {} benchmarks {} times...".format(args.algo, args.num_iteration))
    print("\nSRL-Models:\t{}".format(srl_models))
    print("environments:\t{}".format(envs))
    print("verbose:\t{}".format(args.verbose))
    print("timesteps:\t{}".format(args.num_timesteps))
    for model in srl_models:
        for env in envs:
            for i in range(args.num_iteration):

                printGreen(
                    "\nIteration_num={} (seed: {}), Environment='{}', SRL-Model='{}'".format(i, seeds[i], env, model))

                # redefine the parsed args for rl_baselines.train
                loop_args = ['--srl-model', model, '--seed', str(seeds[i]), '--algo', args.algo, '--env', env,
                             '--num-timesteps', str(int(args.num_timesteps))]

                ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + train_args + loop_args, stdout=stdout)

                if ok != 0:
                    # throw the error down to the terminal
                    raise ChildProcessError("An error occured, error code: {}".format(ok))


if __name__ == '__main__':
    main()
