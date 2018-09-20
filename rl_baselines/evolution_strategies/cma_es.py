import time
import pickle

import cma
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_baselines.base_classes import BaseRLObject
from rl_baselines.utils import createEnvs


def detachToNumpy(tensor):
    """
    gets a pytorch tensor and returns a numpy array
    :param tensor: (pytorch tensor)
    :return: (numpy float)
    """
    # detach means to seperate the gradient from the data of the tensor
    return tensor.to(torch.device('cpu')).detach().numpy()


class CMAESModel(BaseRLObject):
    """
    An implementation of CMA-ES
    CMA-ES: https://pdfs.semanticscholar.org/9b95/6e094c3aa5a831c9b916dde35d1ca9abf066.pdf
    """
    def __init__(self):
        super(CMAESModel, self).__init__()
        self.policy = None
        self.n_population = None
        self.mu = None
        self.sigma = None
        self.continuous_actions = None
        self.es = None
        self.best_model = None

    def save(self, save_path, _locals=None):
        assert self.policy is not None, "Error: must train or load model before use"
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, load_path, args=None):
        with open(load_path, "rb") as f:
            class_dict = pickle.load(f)
        loaded_model = CMAESModel()
        loaded_model.__dict__ = class_dict
        return loaded_model

    def customArguments(self, parser):
        parser.add_argument('--num-population', help='Number of population', type=int, default=20)
        parser.add_argument('--mu', type=float, default=0,
                            help='inital location for gaussian sampling of network parameters')
        parser.add_argument('--sigma', type=float, default=0.14,
                            help='inital scale for gaussian sampling of network parameters')
        parser.add_argument('--cuda', action='store_true', default=False,
                            help='use gpu for the neural network')
        parser.add_argument('--deterministic', action='store_true', default=False,
                            help='do a deterministic approach for the actions on the output of the policy')
        return parser

    def getActionProba(self, observation, dones=None):
        assert self.policy is not None, "Error: must train or load model before use"
        return self.policy.getActionProba(observation)

    def getAction(self, observation, dones=None):
        assert self.policy is not None, "Error: must train or load model before use"
        return self.policy.getAction(observation)

    @classmethod
    def getOptParam(cls):
        return {
            "sigma": (float, (0, 0.2))
        }

    @classmethod
    def makeEnv(cls, args, env_kwargs=None, load_path_normalise=None):
        return createEnvs(args, allow_early_resets=True, env_kwargs=env_kwargs, load_path_normalise=load_path_normalise)

    def train(self, args, callback, env_kwargs=None, train_kwargs=None):
        args.num_cpu = args.num_population
        env = self.makeEnv(args, env_kwargs=env_kwargs)

        args.__dict__.update(train_kwargs)

        if args.continuous_actions:
            action_space = np.prod(env.action_space.shape)
        else:
            action_space = env.action_space.n

        if args.srl_model != "raw_pixels":
            net = MLPPolicyPytorch(np.prod(env.observation_space.shape), [100], action_space)
        else:
            net = CNNPolicyPytorch(env.observation_space.shape[-1], action_space)

        self.policy = PytorchPolicy(net, args.continuous_actions, srl_model=(args.srl_model != "raw_pixels"),
                                    cuda=args.cuda, deterministic=args.deterministic)
        self.n_population = args.num_population
        self.mu = args.mu
        self.sigma = args.sigma
        self.continuous_actions = args.continuous_actions
        self.es = cma.CMAEvolutionStrategy(self.policy.getParamSpace() * [self.mu], self.sigma,
                                           {'popsize': self.n_population})
        self.best_model = np.array(self.policy.getParamSpace() * [self.mu])
        num_updates = int(args.num_timesteps)

        start_time = time.time()
        step = 0
        while step < num_updates:
            obs = env.reset()
            r = np.zeros((self.n_population,))
            # here, CMAEvolutionStrategy will return a list of param for each of the population
            population = self.es.ask()
            done = np.full((self.n_population,), False)
            while not done.all():
                actions = []
                for k in range(self.n_population):
                    if not done[k]:
                        self.policy.setParam(population[k])
                        action = self.policy.getAction(np.array([obs[k]]))[0]
                        actions.append(action)
                    else:
                        actions.append(None)  # do nothing, as we are done

                obs, reward, new_done, info = env.step(actions)
                step += np.sum(~done)

                done = np.bitwise_or(done, new_done)

                # cumulate the reward for every enviroment that is not finished
                r[~done] += reward[~done]

                if callback is not None:
                    callback(locals(), globals())

            print("{} steps - {:.2f} FPS".format(step, step / (time.time() - start_time)))
            self.es.tell(population, -r)
            self.best_model = self.es.result.xbest


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
    :param deterministic: (bool) Do a deterministic approach for the actions on the output of the policy
    """

    def __init__(self, model, continuous_actions, srl_model=True, cuda=False, deterministic=False):
        super(PytorchPolicy, self).__init__(continuous_actions)
        self.model = model
        self.param_len = np.sum([np.prod(x.shape) for x in self.model.parameters()])
        self.continuous_actions = continuous_actions
        self.srl_model = srl_model
        self.cuda = cuda
        self.deterministic = deterministic
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

        self.model = self.model.to(self.device)

    # used to prevent pickling of pytorch device object, as they cannot be pickled
    def __getstate__(self):
        d = self.__dict__.copy()
        d['model'] = d['model'].to(torch.device('cpu'))
        if 'device' in d:
            d['device'] = 'cpu'
        return d

    # restore torch device from a pickle using the same config, if cuda is available
    def __setstate__(self, d):
        if 'device' in d:
            d['device'] = torch.device("cuda" if torch.cuda.is_available() and d['cuda'] else "cpu")
        d['model'] = d['model'].to(d['device'])
        self.__dict__.update(d)

    def getActionProba(self, obs):
        """
        Returns the action probability for the given observation
        :param obs: (numpy float or numpy int)
        :return: (numpy float) the action probability
        """
        if not self.srl_model:
            obs = np.transpose(obs / 255.0, (0, 3, 1, 2))

        if self.continuous_actions:
            action = detachToNumpy(self.model(self.toTensor(obs)))
        else:
            action = detachToNumpy(F.softmax(self.model(self.toTensor(obs)), dim=-1))
        return action

    def getAction(self, obs):
        """
        Returns an action for the given observation
        :param obs: (numpy float or numpy int)
        :return: the action
        """
        if not self.srl_model:
            obs = np.transpose(obs / 255.0, (0, 3, 1, 2))

        with torch.no_grad():
            if self.continuous_actions:
                action = detachToNumpy(self.model(self.toTensor(obs)))
            else:
                action = detachToNumpy(F.softmax(self.model(self.toTensor(obs)), dim=-1))
                if self.deterministic:
                    action = np.argmax(action, axis=1)
                else:
                    action = np.array([np.random.choice(len(a), p=a) for a in action])

        return action

    def toTensor(self, arr):
        """
        Returns a pytorch Tensor object from a numpy array
        :param arr: ([float])
        :return: (Tensor)
        """
        return torch.from_numpy(arr).to(torch.float).to(self.device)

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
        nn.utils.vector_to_parameters(self.toTensor(param).contiguous(), self.model.parameters())


class CNNPolicyPytorch(nn.Module):
    """
    A simple CNN policy using pytorch
    :param out_dim: (int)
    """

    def __init__(self, in_dim, out_dim):
        super(CNNPolicyPytorch, self).__init__()
        # Set bias to False, due to it being nullified and replaced by BatchNorm2d
        self.conv1 = nn.Conv2d(in_dim, 8, kernel_size=5, padding=2, stride=2, bias=False)
        self.norm1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2, bias=False)
        self.norm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2, bias=False)
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
