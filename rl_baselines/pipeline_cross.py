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
from srl_zoo.utils import printGreen, printRed, printYellow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # used to remove debug info of tensorflow


'''
example:
------------------------------------------------------------------------------
python -m rl_baselines.pipeline_cross --algo ppo2 --log-dir logs/ 
--srl-model srl_combination ground_truth --num-iteration 1 --num-timesteps 100000 
--task sc cc 
--srl-config-file config/srl_models.yaml config/srl_models_test.yaml
------------------------------------------------------------------------------
--srl-config-file : a list of config_file which should have the same number as tasks. (or only one, it will take this one for all tasks by default)
'''

def main():
    parser = argparse.ArgumentParser(description="OpenAI RL Baselines Benchmark",
                                     epilog='After the arguments are parsed, the rest are assumed to be arguments for' +
                                            ' rl_baselines.train')
    parser.add_argument('--algo', type=str, default='ppo2', help='OpenAI baseline to use',
                        choices=list(registered_rl.keys()))
    parser.add_argument('--env', type=str, nargs='+', default=["OmnirobotEnv-v0"], help='environment ID(s)',
                        choices=["OmnirobotEnv-v0"])#list(registered_env.keys()))
    parser.add_argument('--srl-model', type=str, nargs='+', default=["ground_truth"],
                        help='SRL model(s) to use',
                        choices=list(registered_srl.keys()))
    parser.add_argument('--num-timesteps', type=int, default=1e6, help='number of timesteps the baseline should run')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display baseline STDOUT')
    parser.add_argument('--num-iteration', type=int, default=15,
                        help='number of time each algorithm should be run for each unique combination of environment ' +
                             ' and srl-model.')
    parser.add_argument('--seed', type=int, default=0,
                        help='initial seed for each unique combination of environment and srl-model.')
    parser.add_argument('--srl-config-file', nargs='+', type=str, default=["config/srl_models.yaml"],
                        help='Set the location of the SRL model path configuration.')
    
    parser.add_argument('--tasks', type=str, nargs='+', default=["cc"],
                        help='The tasks for the robot',
                        choices=["cc","ec","sqc","sc"])
#    parser.add_argument('--srl-modell', type=str, default="",help='')
    # returns the parsed arguments, and the rest are assumed to be arguments for rl_baselines.train
    args, train_args = parser.parse_known_args()



    

    # Sanity check
    assert args.num_timesteps >= 1, "Error: --num-timesteps cannot be less than 1"
    assert args.num_iteration >= 1, "Error: --num-iteration cannot be less than 1"

    # Removing duplicates and sort
    srl_models = list(set(args.srl_model))
    envs = list(set(args.env))
    tasks=list(set(args.tasks))
    tasks.sort()
    srl_models.sort()
    envs.sort()
    tasks=['-'+t  for t in tasks]
    config_files=args.srl_config_file

    # LOAD SRL models list


    if len(config_files)==1:
        printYellow("Your are using the same config file: {} for all training tasks".format(config_files[0]))

        config_files = [config_files[0] for i in range(len(tasks))]
    else:
        assert len(config_files)==len(tasks), \
            "Error:  {} config files given for {} tasks".format(len(config_files),len(tasks))

    for file in config_files:
        assert os.path.exists(file), \
            "Error: cannot load \"--srl-config-file {}\", file not found!".format(file)

    for file in config_files:
        with open(file, 'rb') as f:
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
                printRed("Error: '{}' missing definition for log_folder in environment {}".format(file, env))
                valid = False

            # validate each model for the current env definition
            for model in srl_models:
                if registered_srl[model][0] == SRLType.ENVIRONMENT:
                    continue  # not an srl model, skip to the next model
                elif model not in all_models[env]:
                    printRed("Error: '{}' missing srl_model {} for environment {}".format(file, model, env))
                    valid = False
                elif (not missing_log) and (not os.path.exists(all_models[env]["log_folder"] + all_models[env][model])):
                    # checking presence of srl_model path, if and only if log_folder exists
                    printRed("Error: srl_model {} for environment {} was defined in ".format(model, env) +
                             "'{}', however the file {} it was tagetting does not exist.".format(
                                 file, all_models[env]["log_folder"] + all_models[env][model]))
                    valid = False

        assert valid, "Errors occurred due to malformed {}, cannot continue.".format(file)



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

    num_tasks=len(tasks)


    printGreen("The tasks that will be exacuted: {}".format(args.tasks))
    printGreen("with following config files: {}".format(config_files))


    for model in srl_models:

        for env in envs:
            for iter_task in range(num_tasks):

                for i in range(args.num_iteration):
    
                    printGreen(
                        "\nIteration_num={} (seed: {}), Environment='{}', SRL-Model='{}' , Task='{}', Config_file='{}'".format(i, seeds[i], env, model, tasks[iter_task]),config_files[iter_task])
    
                    # redefine the parsed args for rl_baselines.train
                    loop_args = ['--srl-model', model, '--seed', str(seeds[i]),
                                 '--algo', args.algo, '--env', env,
                                 '--num-timesteps', str(int(args.num_timesteps)), 
                                 '--srl-config-file', config_files[iter_task], tasks[iter_task]]
                    ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + train_args + loop_args, stdout=stdout)
    
                    if ok != 0:
                        # throw the error down to the terminal
                        raise ChildProcessError("An error occured, error code: {}".format(ok))
    

if __name__ == '__main__':
    main()
