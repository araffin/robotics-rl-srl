import time
import pickle

import cma
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

import environments.kuka_button_gym_env as kuka_env
from environments.utils import makeEnv
from rl_baselines.utils import CustomVecNormalize
from srl_priors.utils import printYellow

# my new best friend (signed hill-a)
np.seterr(invalid='raise')

class Policy(object):
    def __init__(self):
        pass

    def getAction(self, obs):
        raise NotImplementedError

    def getParamSpace(self):
        raise NotImplementedError

    def setParam(self):
        raise NotImplementedError


class PytorchPolicy(Policy):
    def __init__(self, model, continuous_actions):
        super(PytorchPolicy, self).__init__()
        self.model = model
        self.param_len = np.sum([np.prod(x.shape) for x in self.model.parameters()])
        self.continuous_actions = continuous_actions

    def getAction(self, obs):
        if self.continuous_actions:
            return self.model(self.make_var(obs.reshape(-1))).data.numpy()
        else:
            return np.argmax(self.model(self.make_var(obs.reshape(-1))).data)

    @staticmethod
    def make_var(arr):
        return Variable(torch.from_numpy(arr))

    def getParamSpace(self):
        return self.param_len

    def setParam(self, param):
        nn.utils.vector_to_parameters(self.make_var(param).contiguous(), self.model.parameters())


class CMAES:
    def __init__(self, n_population, policy, continuous_actions=False):
        self.policy = policy
        self.n_population = n_population
        self.continuous_actions = continuous_actions

    def getAction(self, obs):
        return self.policy.getAction(obs)

    def save(self, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def train(self, env, callback, num_updates=1e6):
        es = cma.CMAEvolutionStrategy(self.policy.getParamSpace() * [0], 1, {'popsize': self.n_population})
        start_time = time.time()
        step = 0

        while(step < num_updates):
            obs = env.reset()
            r = np.zeros((self.n_population,))
            population = es.ask()
            done = np.full((self.n_population,), False)
            while not done.all():
                actions = []
                for k in range(self.n_population):
                    if not done[k]:
                        current_obs = obs[k].reshape(-1)
                        self.policy.setParam(population[k])
                        action = self.getAction(current_obs)
                        actions.append(action)
                    else:
                        actions.append(None) # do nothing, as we are done

                obs, reward, new_done, info = env.step(actions)
                step += self.n_population

                done = np.bitwise_or(done,new_done)

                # cumulate the reward for every enviroment that is not finished
                r[~done] += reward[~done]

                if callback is not None:
                    callback(locals(), globals())
                if (step/self.n_population + 1) % 500 == 0:
                    print("{} steps - {:.2f} FPS".format(step, step / (time.time() - start_time)))

            es.tell(population, -r)



def load(save_path):
    with open(save_path, "rb") as f:
        class_dict = pickle.load(f)
    model = CMAES(1,(1,1))
    model.__dict__ = class_dict
    return model

def customArguments(parser):
    """
    :param parser: (ArgumentParser Object)
    :return: (ArgumentParser Object)
    """
    parser.add_argument('--num-population', help='Number of population', type=int, default=20)
    return parser

def main(args, callback=None):
    """
    :param args: (argparse.Namespace Object)
    :param callback: (function)
    """

    envs = [makeEnv(args.env, args.seed, i, args.log_dir, allow_early_resets=True)
            for i in range(args.num_population)]
    envs = SubprocVecEnv(envs)
    envs = VecFrameStack(envs, args.num_stack)

    if args.srl_model != "":
        printYellow("Using MLP policy because working on state representation")
        args.policy = "mlp"
        envs = CustomVecNormalize(envs, norm_obs=True, norm_rewards=False)

    if args.continuous_actions:
        action_space = np.prod(envs.action_space.shape)
    else:
        action_space = envs.action_space.n

    policy = PytorchPolicy(Net(np.prod(envs.observation_space.shape), 100, action_space, args.continuous_actions), args.continuous_actions)

    model = CMAES(
        args.num_population, 
        policy, 
        continuous_actions=args.continuous_actions
    )

    model.train(envs, callback, num_updates=(int(args.num_timesteps) // args.num_population*2))


class Net(nn.Module):
    def __init__(self, in_dim, lin_dim, out_dim, continuous_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(int(in_dim), int(lin_dim))
        self.fc2 = nn.Linear(int(lin_dim), int(out_dim))

        self.continuous_actions = continuous_actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if self.continuous_actions:
            return x
        else:
            return F.softmax(x, dim=-1)
