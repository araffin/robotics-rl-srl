import argparse
import subprocess
import os
import shutil
import glob
import pprint
import math
import time

import pandas as pd
import numpy as np

from rl_baselines.registry import registered_rl
from environments.registry import registered_env
from state_representation.registry import registered_srl
from srl_zoo.utils import printGreen

ITERATION_SCALE = 10000
MIN_ITERATION = 30000


class Hyperband(object):
    def __init__(self, param_sampler, train, max_iter=100, eta=3.0):
        self.param_sampler = param_sampler
        self.train = train
        self.max_iter = max_iter
        self.eta = eta
        self.s_max = int(math.floor(math.log(self.max_iter) / math.log(self.eta)))
        self.B = (self.s_max + 1) * self.max_iter

        self.history = []

    def run(self):
        for s in reversed(range(self.s_max + 1)):
            n = int(math.ceil(self.B / self.max_iter * self.eta**s / (s + 1)))
            r = self.max_iter * self.eta**(-s)

            all_parameters = np.array([self.param_sampler() for _ in range(n)])
            for i in range(s+1):
                printGreen("\npop_itt:{}/{}, itt:{}/{}, pop_size:{}".format(s, self.s_max + 1, i, s+1,
                                                                            len(all_parameters)))
                n_i = int(math.floor(n * self.eta**(-i)))
                r_i = r * self.eta**i
                losses = [self.train(t, r_i, k) for k, t in enumerate(all_parameters)]

                self.history.extend(zip([(t, r_i) for t in all_parameters], losses))
                all_parameters = all_parameters[np.argsort(losses)[:int(math.floor(n_i / self.eta))]]

        return self.history[int(np.argmin([val[1] for val in self.history]))]


def main():
    parser = argparse.ArgumentParser(description="OpenAI RL Baselines")
    parser.add_argument('--algo', default='ppo2', choices=list(registered_rl.keys()), help='OpenAI baseline to use',
                        type=str)
    parser.add_argument('--env', type=str, help='environment ID', default='KukaButtonGymEnv-v0',
                        choices=list(registered_env.keys()))
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--srl-model', type=str, default='raw_pixels', choices=list(registered_srl.keys()),
                        help='SRL model to use')
    parser.add_argument('--num-timesteps', type=int, default=1e6, help='number of timesteps the baseline should run')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display baseline STDOUT')
    parser.add_argument('--max_iter', type=int, default=100, help='Number of iteration to try')

    args, train_args = parser.parse_known_args()

    train_args.extend(['--srl-model', args.srl_model, '--seed', str(args.seed), '--algo', args.algo, '--env', args.env,
                       '--log-dir', "logs/_hyperband_search/", '--no-vis'])

    if args.verbose:
        # None here means stdout of terminal for subprocess.call
        stdout = None
    else:
        stdout = open(os.devnull, 'w')

    opt_param = registered_rl[args.algo][0].getOptParam()
    if opt_param is None:
        raise AssertionError("Error: {} algo does not support Hyperband search.".format(args.algo))

    rng = np.random.RandomState(args.seed)

    def sample():
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

    def train(params, num_iters, train_id):
        printGreen("\nID_num={}, Num-timesteps={}, Param:"
                   .format(train_id, int(max(MIN_ITERATION, num_iters * ITERATION_SCALE))))
        pprint.pprint(params)

        # cleanup old files
        if os.path.exists("logs/_hyperband_search/"):
            shutil.rmtree("logs/_hyperband_search/")

        loop_args = ['--num-timesteps', str(int(max(MIN_ITERATION, num_iters * ITERATION_SCALE)))]

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

        if np.isnan(rewards).any():
            rewards = -np.inf

        print("reward: ", np.mean(rewards))

        return -np.mean(rewards)

    opt = Hyperband(sample, train, max_iter=args.num_timesteps // ITERATION_SCALE)
    t_start = time.time()
    opt.run()
    all_params, loss = zip(*opt.history)
    idx = np.argmin(loss)
    opt_params, nb_iter = all_params[idx]
    reward = loss[idx]
    print('time to run : {}s'.format(int(time.time() - t_start)))
    print('Total nb. evaluations : {}'.format(len(all_params)))
    print('Best nb. of iterations : {}'.format(int(nb_iter)))
    print('Best params : {}'.format(opt_params))
    print('Best reward : {:.3f}'.format(-reward))

    param_dict, timesteps = zip(*all_params)
    output = pd.DataFrame(list(param_dict))
    output["timesteps"] = np.array(np.maximum(MIN_ITERATION, np.array(timesteps) * ITERATION_SCALE).astype(int))
    output["reward"] = -np.array(loss)
    output.to_csv("logs/hyperband_{}_{}_{}_seed{}_numtimestep{}.csv"
                  .format(args.algo, args.env, args.srl_model, args.seed, args.num_timesteps))


if __name__ == '__main__':
    main()
