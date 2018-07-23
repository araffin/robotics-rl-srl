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
import hyperopt

from rl_baselines.registry import registered_rl
from environments.registry import registered_env
from state_representation.registry import registered_srl
from srl_zoo.utils import printGreen

ITERATION_SCALE = 10000
MIN_ITERATION = 30000


class HyperParameterOptimizer(object):
    def __init__(self, opt_param, train, seed=0):
        """
        the base class for hyper parameter optimizer

        :param opt_param: (dict) the parameters to optimize
        :param train: (function (dict, int, int): float) the function that take:

            - params: (dict) the hyper parameters to train with
            - num_iters (int) the number of itterations to train (can be None)
            - train_id: (int) the training number (can be None)
            - returns: (float) the score of the training to minimize

        :param seed: (int) the initial seed for the random number generator
        """
        self.opt_param = opt_param
        self.train = train
        self.seed = seed

        self.history = []

    def run(self):
        """
        run the hyper parameter search
        """
        raise NotImplementedError


class Hyperband(HyperParameterOptimizer):
    def __init__(self, opt_param, train, seed=0, max_iter=100, eta=3.0):
        """
        A Hyperband implementation, it is similar to a targeted random search

        Hyperband: https://arxiv.org/abs/1603.06560

        :param opt_param: (dict) the parameters to optimize
        :param train: (function (dict, int, int): float) the function that take:

            - params: (dict) the hyper parameters to train with
            - num_iters (int) the number of itterations to train (can be None)
            - train_id: (int) the training number (can be None)
            - returns: (float) the score of the training to minimize

        :param seed: (int) the initial seed for the random number generator
        :param max_iter: (int) the maximum budget for hyperband's search
        :param eta: (float) the reduction factor of the search
        """
        super(Hyperband, self).__init__(opt_param, train, seed=seed)
        self.max_iter = max_iter
        self.eta = eta
        self.max_steps = int(math.floor(math.log(self.max_iter) / math.log(self.eta)))
        self.budget = (self.max_steps + 1) * self.max_iter

        self.rng = np.random.RandomState(seed)
        self.param_sampler = self._generate_sampler()

    def _generate_sampler(self):
        # will generate a hyperparameter sampler for Hyperband
        def _sample():
            params = {}
            for name, (param_type, val) in self.opt_param.items():
                if param_type == int:
                    params[name] = self.rng.randint(val[0], val[1])
                elif param_type == float:
                    params[name] = self.rng.uniform(val[0], val[1])
                elif isinstance(param_type, tuple) and param_type[0] == list:
                    params[name] = val[self.rng.randint(len(val))]
                else:
                    raise AssertionError("Error: unknown type {}".format(param_type))

            return params
        return _sample

    def run(self):
        for step in reversed(range(self.max_steps + 1)):
            max_n_param_sampled = int(math.ceil(self.budget / self.max_iter * self.eta**step / (step + 1)))
            max_iters = self.max_iter * self.eta**(-step)

            all_parameters = np.array([self.param_sampler() for _ in range(max_n_param_sampled)])
            for i in range(step + 1):
                printGreen("\npop_itt:{}/{}, itt:{}/{}, pop_size:{}".format(self.max_steps - step, self.max_steps + 1,
                                                                            i, step+1, len(all_parameters)))
                n_param_sampled = int(math.floor(max_n_param_sampled * self.eta**(-i)))
                num_iters = max_iters * self.eta**i
                losses = [self.train(params, num_iters, train_id) for train_id, params in enumerate(all_parameters)]

                self.history.extend(zip([(params, num_iters) for params in all_parameters], losses))
                all_parameters = all_parameters[np.argsort(losses)[:int(math.floor(n_param_sampled / self.eta))]]

        return self.history[int(np.argmin([val[1] for val in self.history]))]


class Hyperopt(HyperParameterOptimizer):
    def __init__(self, opt_param, train, seed=0, num_eval=100):
        """
        A Hyperopt implementation, it is similar to a bayesian search

        Hyperopt: https://www.lri.fr/~kegl/research/PDFs/BeBaBeKe11.pdf

        :param opt_param: (dict) the parameters to optimize
        :param train: (function (dict, int, int): float) the function that take:

            - params: (dict) the hyper parameters to train with
            - num_iters (int) the number of itterations to train (can be None)
            - train_id: (int) the training number (can be None)
            - returns: (float) the score of the training to minimize

        :param seed: (int) the initial seed for the random number generator
        :param num_eval: (int) the number of evaluation to do
        """
        super(Hyperopt, self).__init__(opt_param, train, seed=seed)
        self.num_eval = num_eval
        self.search_space = []
        for name, (param_type, val) in self.opt_param.items():
            if param_type == int:
                self.search_space.append(hyperopt.hp.quniform(name, val[0], val[1], 1))
            elif param_type == float:
                self.search_space.append(hyperopt.hp.uniform(name, val[0], val[1]))
            elif isinstance(param_type, tuple) and param_type[0] == list:
                self.search_space.append(hyperopt.hp.choice(name, val))
            else:
                raise AssertionError("Error: unknown type {}".format(param_type))

    def run(self):
        trials = hyperopt.Trials()
        hyperopt.fmin(lambda **kwargs: {'loss': self.train(kwargs), 'status': hyperopt.STATUS_OK},
                      space=self.search_space,
                      algo=hyperopt.tpe.suggest,
                      max_evals=self.num_eval,
                      trials=trials)
        self.history.extend(zip(trials.trials, trials.losses()))
        return self.history[int(np.argmin([val[1] for val in self.history]))]


def makeRlTrainingFunction(args, train_args):
    """
    makes a training function for the hyperparam optimizers

    :param args: (ArgumentParser) the optimizer arguments
    :param train_args: (ArgumentParser) the remaining arguments
    :return: (function (dict, int, int): float) the function that take:

        - params: (dict) the hyper parameters to train with
        - num_iters (int) the number of iterations to train (can be None)
        - train_id: (int) the current iteration number in the hyperparameter search (can be None)
        - returns: (float) the score of the training to minimize
    """
    if args.verbose:
        # None here means stdout of terminal for subprocess.call
        stdout = None
    else:
        stdout = open(os.devnull, 'w')

    def _train(params, num_iters=None, train_id=None):
        # generate a print string
        print_str = "\n"
        format_args = []
        if train_id is not None:
            print_str += "ID_num={}, "
            format_args.append(train_id)
        if num_iters is not None:
            print_str += "Num-timesteps={}, "
            format_args.append(int(max(MIN_ITERATION, num_iters * ITERATION_SCALE)))

        print_str += "Param:"
        printGreen(print_str.format(*format_args))
        pprint.pprint(params)

        # cleanup old files
        if os.path.exists("logs/_hyperband_search/"):
            shutil.rmtree("logs/_hyperband_search/")

        # add the training args that where parsed for the hyperparam optimizers
        if num_iters is not None:
            loop_args = ['--num-timesteps', str(int(max(MIN_ITERATION, num_iters * ITERATION_SCALE)))]
        else:
            loop_args = ['--num-timesteps', str(int(args.num_timesteps))]

        # redefine the hyperparam args for rl_baselines.train
        if len(params) > 0:
            loop_args.append("--hyperparam")
            for param_name, param_val in params.items():
                loop_args.append("{}:{}".format(param_name, param_val))

        # call the training
        ok = subprocess.call(['python', '-m', 'rl_baselines.train'] + train_args + loop_args, stdout=stdout)
        if ok != 0:
            # throw the error down to the terminal
            raise ChildProcessError("An error occured, error code: {}".format(ok))

        # load the logging of the training, and extract the reward
        folders = glob.glob("logs/_hyperband_search/{}/{}/{}/*".format(args.env, args.srl_model, args.algo))
        assert len(folders) != 0, "Error: Could not find generated directory, halting hyperband search."
        rewards = []
        for monitor_path in glob.glob(folders[0] + "/*.monitor.csv"):
            rewards.append(np.mean(pd.read_csv(monitor_path, skiprows=1)["r"][-10:]))
        if np.isnan(rewards).any():
            rewards = -np.inf
        print("reward: ", np.mean(rewards))

        # negative reward, as we are minimizing with hyperparameter search
        return -np.mean(rewards)
    return _train


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for implemented RL models")
    parser.add_argument('--optimizer', default='hyperband', choices=['hyperband', 'hyperopt'], type=str,
                        help='The hyperparameter optimizer to choose from')
    parser.add_argument('--algo', default='ppo2', choices=list(registered_rl.keys()), help='OpenAI baseline to use',
                        type=str)
    parser.add_argument('--env', type=str, help='environment ID', default='KukaButtonGymEnv-v0',
                        choices=list(registered_env.keys()))
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--srl-model', type=str, default='raw_pixels', choices=list(registered_srl.keys()),
                        help='SRL model to use')
    parser.add_argument('--num-timesteps', type=int, default=1e6, help='number of timesteps the baseline should run')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Display baseline STDOUT')
    parser.add_argument('--max-eval', type=int, default=100, help='Number of evalutation to try for hyperopt')

    args, train_args = parser.parse_known_args()

    train_args.extend(['--srl-model', args.srl_model, '--seed', str(args.seed), '--algo', args.algo, '--env', args.env,
                       '--log-dir', "logs/_hyperband_search/", '--no-vis'])

    # verify the algorithm has defined it, and that it returnes an expected value
    try:
        opt_param = registered_rl[args.algo][0].getOptParam()
        assert opt_param is not None
    except AttributeError or AssertionError:
        raise AssertionError("Error: {} algo does not support Hyperband search.".format(args.algo))

    if args.optimizer == "hyperband":
        opt = Hyperband(opt_param, makeRlTrainingFunction(args, train_args), seed=args.seed,
                        max_iter=args.num_timesteps // ITERATION_SCALE)
    elif args.optimizer == "hyperopt":
        opt = Hyperopt(opt_param, makeRlTrainingFunction(args, train_args), seed=args.seed, num_eval=args.max_eval)
    else:
        raise ValueError("Error: optimizer {} was defined but not implemented, Halting.".format(args.optimizer))

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
