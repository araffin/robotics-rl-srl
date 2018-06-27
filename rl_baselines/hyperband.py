import argparse
import subprocess
import os
import shutil
import glob
import pprint

from fluentopt.hyperband import hyperband
import pandas as pd
import numpy as np

from rl_baselines.registry import registered_rl
from environments.registry import registered_env
from state_representation.registry import registered_srl
from srl_zoo.utils import printGreen


def main():
    parser = argparse.ArgumentParser(description="OpenAI RL Baselines")
    parser.add_argument('--algo', default='ppo2', choices=list(registered_rl.keys()), help='OpenAI baseline to use',
                        type=str)
    parser.add_argument('--env', type=str, help='environment ID', default='KukaButtonGymEnv-v0',
                        choices=list(registered_env.keys()))
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--srl-model', type=str, default='raw_pixels', choices=list(registered_srl.keys()),
                        help='SRL model to use')
    parser.add_argument('--num-timesteps', type=int, default=2e5, help='number of timesteps the baseline should run')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display baseline STDOUT')
    parser.add_argument('--max_iter', type=int, default=100, help='Number of iteration to try')

    args, train_args = parser.parse_known_args()

    train_args.extend(['--srl-model', args.srl_model, '--seed', str(args.seed), '--algo', args.algo, '--env', args.env,
                       '--log-dir', "logs/_hyperband_search/", '--num-timesteps', str(int(args.num_timesteps)),
                       '--no-vis'])

    if args.verbose:
        # None here means stdout of terminal for subprocess.call
        stdout = None
    else:
        stdout = open(os.devnull, 'w')

    opt_param = registered_rl[args.algo][0].getOptParam()
    if opt_param is None:
        raise AssertionError("Error: {} algo does not support Hyperband search.".format(args.algo))

    def sample(rng):
        params = {}
        for name, (param_type, val) in opt_param.items():
            if param_type == int:
                params[name] = rng.randint(val[0], val[1])
            elif param_type == float:
                params[name] = rng.uniform(val[0], val[1])
            elif param_type == list:
                params[name] = val[rng.randint(len(val))]
            else:
                raise AssertionError("Error: unknown type {}".format(param_type))

        return params

    def run_batch(batch):
        for i, (num_iters, params) in enumerate(batch):
            printGreen("\nIteration_num={}, Param:".format(i))
            pprint.pprint(params)
            print()

            # cleanup old files
            if os.path.exists("logs/_hyperband_search/"):
                shutil.rmtree("logs/_hyperband_search/")

            loop_args = []

            # redefine the parsed args for rl_baselines.train
            if len(params) > 0:
                loop_args.append("--hyperparam")
                for param_name, param_val in params.items():
                    loop_args.append("{}:{}".format(param_name, param_val))

            ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + train_args + loop_args, stdout=stdout)

            if ok != 0:
                # throw the error down to the terminal
                raise ChildProcessError("An error occured, error code: {}".format(ok))

            folders = glob.glob("logs/_hyperband_search/{}/{}/{}/*".format(args.env, args.srl_model, args.algo))
            assert len(folders) != 0, "Error: Could not find generated directory, halting hyperband search."
            rewards = []
            for montior_path in glob.glob(folders[0] + "/*.monitor.csv"):
                rewards.append(np.mean(pd.read_csv(montior_path, skiprows=1)["r"][-10:]))

            yield -np.mean(rewards)

    input_hist, output_hist = hyperband(sample, run_batch, max_iter=100, random_state=args.seed)
    idx = np.argmin(output_hist)
    nb_iter, opt_params = input_hist[idx]
    reward = output_hist[idx]
    print('Total nb. evaluations : {}'.format(len(input_hist)))
    print('Best nb. of iterations : {}'.format(int(nb_iter)))
    print('Best params : {}'.format(opt_params))
    print('Best reward : {:.3f}'.format(-reward))


if __name__ == '__main__':
    main()
