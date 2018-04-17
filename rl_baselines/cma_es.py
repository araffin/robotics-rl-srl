import time
import pickle

import cma
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import environments.kuka_button_gym_env as kuka_env
from rl_baselines.utils import createEnvs
from srl_priors.utils import printYellow


class Policy(object):
    """
    The policy object for genetic algorithms
    :param continuous_actions: (bool)
    """

    def __init__(self, continuous_actions, srl_model=True):
        self.continuous_actions = continuous_actions
        self.srl_model = srl_model

    def getAction(self, obs):
        raise NotImplementedError

    def getParamSpace(self):
        raise NotImplementedError

    def setParam(self, param):
        raise NotImplementedError


class PytorchPolicy(Policy):
    """
    The policy object for genetic algorithms, using Pytorch networks
    :param model: (Pytorch nn.Module) make sure there is no Sequential, as it breaks .shape function
    :param continuous_actions: (bool)
    :param srl_model: (bool) if using an srl model or not
    :param cuda: (bool)
    """

    def __init__(self, model, continuous_actions, srl_model=True, cuda=False):
        super(PytorchPolicy, self).__init__(continuous_actions)
        self.model = model
        self.param_len = np.sum([np.prod(x.shape) for x in self.model.parameters()])
        self.continuous_actions = continuous_actions
        self.srl_model = srl_model
        self.cuda = cuda

        if self.cuda:
            self.model.cuda()

    def getAction(self, obs):
        """
        Returns an action for the given observation
        :param obs: ([float])
        :return: the action
        """
        if not self.srl_model:
            obs = obs.reshape((1,) + obs.shape)
            obs = np.transpose(obs / 255.0, (0, 3, 1, 2))

        if self.continuous_actions:
            return self.model(self.make_var(obs)).data.numpy()
        else:
            return np.argmax(F.softmax(self.model(self.make_var(obs)), dim=-1).data)

    def make_var(self, arr):
        """
        Returns a pytorch Variable object from a numpy array
        :param arr: ([float])
        :return: (Variable)
        """
        if self.cuda:
            return Variable(torch.from_numpy(arr)).float().cuda()
        else:
            return Variable(torch.from_numpy(arr)).float()

    def getParamSpace(self):
        """
        Returns the size of the parameters for the pytorch network
        :return: (int)
        """
        return self.param_len

    def setParam(self, param):
        """
        Set the network bias and weights
        :param param: ([float])
        """
        nn.utils.vector_to_parameters(self.make_var(param).contiguous(), self.model.parameters())


class CNNPolicyPytorch(nn.Module):
    """
    A simple CNN policy using pytorch
    :param out_dim: (int)
    """

    def __init__(self, out_dim):
        super(CNNPolicyPytorch, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2, stride=2)
        self.norm1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2)
        self.norm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
        self.norm3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2)

        self.fc = nn.Linear(288, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MLPPolicyPytorch(nn.Module):
    """
    A simple MLP policy using pytorch
    :param in_dim: (int)
    :param hidden_dims: ([int])
    :param out_dim: (int)
    """

    def __init__(self, in_dim, hidden_dims, out_dim):
        super(MLPPolicyPytorch, self).__init__()
        self.fc_hidden_name = []

        self.fc_in = nn.Linear(int(in_dim), int(hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.add_module("fc_" + str(i), nn.Linear(int(hidden_dims[i]), int(hidden_dims[i + 1])))
            self.fc_hidden_name.append("fc_" + str(i))
        self.fc_out = nn.Linear(int(hidden_dims[-1]), int(out_dim))

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for name in self.fc_hidden_name:
            x = F.relu(getattr(self, name)(x))
        x = self.fc_out(x)
        return x


class CMAES:
    """
    An implementation of the CMA-ES algorithme
    :param n_population: (int)
    :param policy: (Policy Object)
    :param mu: (float) default=0
    :param sigma: (float) default=1
    :param continuous_actions: (bool) default=False
    """

    def __init__(self, n_population, policy, mu=0, sigma=1, continuous_actions=False):
        self.policy = policy
        self.n_population = n_population
        self.init_mu = mu
        self.init_sigma = sigma
        self.continuous_actions = continuous_actions
        self.es = cma.CMAEvolutionStrategy(self.policy.getParamSpace() * [mu], sigma, {'popsize': n_population})
        self.best_model = self.es.result.xbest

    def getAction(self, obs):
        """
        Returns an action for the given observation
        :param obs: ([float])
        :return: the action
        """
        return self.policy.getAction(obs)

    def save(self, save_path):
        """
        :param save_path: (str)
        """
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def train(self, env, callback, num_updates=1e6):
        """
        :param env: (gym enviroment)
        :param callback: (function)
        :param num_updates: (int) the number of updates to do (default=100000)
        """
        start_time = time.time()
        step = 0

        while step < num_updates:
            obs = env.reset()
            r = np.zeros((self.n_population,))
            population = self.es.ask()
            done = np.full((self.n_population,), False)
            while not done.all():
                actions = []
                for k in range(self.n_population):
                    if not done[k]:
                        current_obs = obs[k].reshape(-1)
                        self.policy.setParam(population[k])
                        action = self.policy.getAction(obs[k])
                        actions.append(action)
                    else:
                        actions.append(None)  # do nothing, as we are done

                obs, reward, new_done, info = env.step(actions)
                step += self.n_population

                done = np.bitwise_or(done, new_done)

                # cumulate the reward for every enviroment that is not finished
                r[~done] += reward[~done]

                if callback is not None:
                    callback(locals(), globals())

            print("{} steps - {:.2f} FPS".format(step, step / (time.time() - start_time)))
            self.es.tell(population, -r)
            self.best_model = self.es.result.xbest


def load(save_path):
    """
    :param save_path: (str)
    :return: (CMAES Object)
    """
    with open(save_path, "rb") as f:
        class_dict = pickle.load(f)
    model = CMAES(class_dict["n_population"], class_dict["policy"], class_dict["init_mu"], class_dict["init_sigma"])
    model.__dict__ = class_dict
    return model


def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--num-population', help='Number of population', type=int, default=20)
    parser.add_argument('--mu', type=float, default=0,
                        help='inital location for gaussian sampling of network parameters')
    parser.add_argument('--sigma', type=float, default=0.2,
                        help='inital scale for gaussian sampling of network parameters')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use gpu for the neural network')
    return parser


def main(args, callback=None):
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    """
    args.num_cpu = args.num_population
    envs = createEnvs(args, allow_early_resets=True)

    if args.continuous_actions:
        action_space = np.prod(envs.action_space.shape)
    else:
        action_space = envs.action_space.n

    if args.srl_model != "":
        printYellow("Using MLP policy because working on state representation")
        args.policy = "mlp"
        net = MLPPolicyPytorch(np.prod(envs.observation_space.shape), [100], action_space)
    else:
        net = CNNPolicyPytorch(action_space)

    policy = PytorchPolicy(net, args.continuous_actions, srl_model=(args.srl_model != ""), cuda=args.cuda)

    model = CMAES(
        args.num_population,
        policy,
        mu=args.mu,
        sigma=args.sigma,
        continuous_actions=args.continuous_actions
    )

    model.train(envs, callback, num_updates=int(args.num_timesteps))
