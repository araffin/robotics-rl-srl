import time
import pickle
import math
import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal, Uniform
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from environments.utils import makeEnv
from rl_baselines.base_classes import BaseRLObject
from rl_baselines.utils import CustomVecNormalize, CustomDummyVecEnv, WrapFrameStack, \
    loadRunningAverage, MultiprocessSRLModel
from srl_zoo.utils import printYellow

def l2Loss(tensor):
    """
    L2 loss given a tensor
    :param tensor: (th.Tensor)
    :return: (th.Tensor)
    """
    return (tensor.float() ** 2).mean()

def encodeOneHot(tensor, n_dim):
    """
    One hot encoding for a given tensor
    :param tensor: (th Tensor)
    :param n_dim: (int) Number of dimensions
    :return: (th.Tensor)
    """
    encoded_tensor = th.Tensor(tensor.shape[0], n_dim).zero_().to(tensor.device)
    return encoded_tensor.scatter_(1, tensor, 1.)


def toTensor(arr, device):
    """
    Returns a pytorch Tensor object from a numpy array
    :param arr: ([float])
    :return: (Tensor)
    """
    return th.from_numpy(arr).to(th.float).to(device)


def detachToNumpy(tensor):
    """
    Gets a pytorch tensor and returns a numpy array
    :param tensor: (th.Tensor)
    :return: (numpy float)
    """
    # detach means to seperate the gradient from the data of the tensor
    return tensor.to(th.device('cpu')).detach().numpy()


def softUpdate(*, source, target, factor):
    """
    Update (softly) the weights of target network towards the weights of a source network.
    The amount of change is regulated by a factor.
    :param source: (Pytorch Model)
    :param target: (Pytorch Model)
    :param factor: (float) soft update factor in [0, 1]
    """
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(
            source_param.data * factor + target_param.data * (1.0 - factor)
        )


def hardUpdate(*, source, target):
    """
    Copy the weights from source network to target network
    :param source: (Pytorch Model)
    :param target: (Pytorch Model)
    """
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(source_param.data)


class SAC2Model(BaseRLObject):
    """
    Class containing an implementation of soft actor critic
    """

    def __init__(self):
        super(SAC2Model, self).__init__()

    def save(self, save_path, _locals=None):
        pass

    @classmethod
    def load(cls, load_path, args=None):
        pass

    @classmethod
    def makeEnv(cls, args, env_kwargs=None, load_path_normalise=None):
        # Even though DeepQ is single core only, we need to use the pipe system to work
        if env_kwargs is not None and env_kwargs.get("use_srl", False):
            srl_model = MultiprocessSRLModel(1, args.env, env_kwargs)
            env_kwargs["state_dim"] = srl_model.state_dim
            env_kwargs["srl_pipe"] = srl_model.pipe

        env = CustomDummyVecEnv([makeEnv(args.env, args.seed, 0, args.log_dir, env_kwargs=env_kwargs)])

        if args.srl_model != "raw_pixels":
            env = CustomVecNormalize(env, norm_obs=True, norm_rewards=False)
            env = loadRunningAverage(env, load_path_normalise=load_path_normalise)

        # Normalize only raw pixels
        # WARNING: when using framestacking, the memory used by the replay buffer can grow quickly
        return WrapFrameStack(env, args.num_stack, normalize=args.srl_model == "raw_pixels")

    def customArguments(self, parser):
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='Disable cuda for the neural network')
        parser.add_argument('--buffer-size', type=int, default=int(1e3), help="Replay buffer size")
        parser.add_argument('--reward-scale', type=float, default=int(1), help="Scaling factor for raw reward. (entropy factor)")
        parser.add_argument('--deterministic', action='store_true', default=False,
                            help='deterministic policy for the actions on the output of the policy')
        return parser

    def getActionProba(self, observation, dones=None):
        pass

    def getAction(self, observation, dones=None):
        pass

    def train(self, args, callback, env_kwargs=None):
        env = self.makeEnv(args, env_kwargs=env_kwargs)
        global device
        device = th.device("cuda" if th.cuda.is_available() and not args.no_cuda else "cpu")

        action_dim = env.action_space.shape[0]
        state_dim  = env.observation_space.shape[0]
        hidden_dim = 256

        value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
        target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

        soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
            target_param.data.copy_(param.data)


        value_criterion  = nn.MSELoss()
        soft_q_criterion = nn.MSELoss()

        value_lr  = 3e-4
        soft_q_lr = 3e-4
        policy_lr = 3e-4

        value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
        soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
        policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


        replay_buffer_size = 1000000
        # replay_buffer = ReplayBuffer(replay_buffer_size)
        replay_buffer = ReplayBuffer(replay_buffer_size)

        def soft_q_update(batch_size,
                   gamma=0.99,
                   mean_l=1e-3,
                   std_l=1e-3,
                   z_l=0.0,
                   soft_tau=1e-2):
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            # state, action, reward, next_state, done = map(lambda x: toTensor(x, device),
            #                                                          replay_buffer.sample(batch_size))

            state      = th.FloatTensor(state).to(device)
            next_state = th.FloatTensor(next_state).to(device)
            action     = th.FloatTensor(action).to(device)
            reward     = th.FloatTensor(reward).unsqueeze(1).to(device)
            done       = th.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
            # state = state.float()
            # next_state = state.float()
            # action = action.float()
            # reward = reward.float().unsqueeze(1)
            # done = done.float().unsqueeze(1)

            expected_q_value = soft_q_net(state, action)
            expected_value   = value_net(state)
            new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)


            target_value = target_value_net(next_state)
            next_q_value = reward + (1 - done) * gamma * target_value
            q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

            expected_new_q_value = soft_q_net(state, new_action)
            next_value = expected_new_q_value - log_prob
            value_loss = value_criterion(expected_value, next_value.detach())

            log_prob_target = expected_new_q_value - expected_value
            policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()


            mean_loss = mean_l * mean.pow(2).mean()
            std_loss  = std_l  * log_std.pow(2).mean()
            z_loss    = z_l    * z.pow(2).sum(1).mean()

            policy_loss += mean_loss + std_loss + z_loss

            soft_q_optimizer.zero_grad()
            q_value_loss.backward()
            soft_q_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()


            for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

        max_steps   = 500
        frame_idx   = 0
        batch_size  = 128

        while frame_idx < args.num_timesteps:
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                action = policy_net.get_action(state)
                next_state, reward, done, _ = env.step(action)

                if callback is not None:
                    callback(locals(), globals())

                # replay_buffer.add(state, action, reward, next_state, float(done))
                replay_buffer.push(state, action, reward, next_state, done)
                if len(replay_buffer) > batch_size:
                    soft_q_update(batch_size)

                state = next_state
                episode_reward += reward
                frame_idx += 1

                if done:
                    break



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = th.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = th.tanh(z)

        log_prob = normal.log_prob(z) - th.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std


    def get_action(self, state):
        state = th.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z      = normal.sample()
        action = th.tanh(z)

        action  = action.detach().cpu().numpy()
        return action[0]
