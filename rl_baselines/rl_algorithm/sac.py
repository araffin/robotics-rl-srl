import time
import pickle
import random
import itertools

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Uniform
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from environments.utils import makeEnv
from rl_baselines.base_classes import BaseRLObject
from rl_baselines.utils import CustomVecNormalize, CustomDummyVecEnv, WrapFrameStack, \
    loadRunningAverage, MultiprocessSRLModel
from rl_baselines.models.sac_models import *
from srl_zoo.utils import printYellow


def l2Loss(tensor):
    """
    L2 loss given a tensor
    :param tensor: (th.Tensor)
    :return: (th.Tensor)
    """
    return (tensor.float() ** 2).mean()


def toTensor(arr, device):
    """
    Returns a pytorch Tensor object from a numpy array
    :param arr: (numpy array)
    :return: (Tensor)
    """
    return th.from_numpy(arr).to(device)


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


class SACModel(BaseRLObject):
    """
    Class containing an implementation of soft actor critic
    """

    def __init__(self):
        super(SACModel, self).__init__()

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

        self.device = th.device("cuda" if th.cuda.is_available() and not args.no_cuda else "cpu")
        self.using_images = args.srl_model == "raw_pixels"

        if args.continuous_actions:
            action_space = np.prod(env.action_space.shape)
        else:
            action_space = env.action_space.n

        if args.srl_model != "raw_pixels":
            printYellow("Using MLP policy because working on state representation")
            args.policy = "mlp"
            input_dim = np.prod(env.observation_space.shape)
            # Hack to prevent from duplicated code
            # self.encoder_net = Identity().to(self.device)
        else:
            n_channels = env.observation_space.shape[-1]
            self.encoder_net = NatureCNN(n_channels).to(self.device)
            # self.encoder_net = CustomCNN(n_channels).to(self.device)
            input_dim = 512  # output dim of the encoder net


        self.policy_net = MLPPolicy(input_dim, action_space).to(self.device)
        self.q_value_net = MLPQValueNetwork(input_dim, action_space, args.continuous_actions).to(self.device)
        self.value_net = MLPValueNetwork(input_dim).to(self.device)
        self.target_value_net = MLPValueNetwork(input_dim).to(self.device)

        self.policy = PytorchPolicy(self.policy_net, args.continuous_actions, device=self.device, srl_model=(args.srl_model != "raw_pixels"),
                                    stochastic=not args.deterministic)
        # Make sure target net has the same weights
        hardUpdate(source=self.value_net, target=self.target_value_net)

        value_criterion  = nn.MSELoss()
        q_value_criterion = nn.MSELoss()

        gamma = 0.99
        learning_rate = 3e-4
        # mixture_components = 4
        gradient_steps = 1
        print_freq = 500
        batch_size = 128
        soft_update_factor = 1e-2
        w_reg = 1e-3
        learn_start = 0
        replay_buffer = ReplayBuffer(args.buffer_size)

        optimizer = th.optim.Adam(itertools.chain(self.q_value_net.parameters(), self.value_net.parameters(), self.q_value_net.parameters()), lr=learning_rate)

        obs = env.reset()
        start_time = time.time()

        for step in range(args.num_timesteps):
            obs_tensor = toTensor(obs[None], self.device).float()
            if self.using_images:
                with th.no_grad():
                    obs_tensor = channelFirst(obs_tensor)
                    obs_tensor = self.encoder_net(obs_tensor)
            action = self.policy.getAction(obs_tensor)[0]
            new_obs, reward, done, info = env.step(action)

            replay_buffer.add(obs, action, reward, new_obs, float(done))
            obs = new_obs

            if callback is not None:
                callback(locals(), globals())

            if done:
                obs = env.reset()

            for _ in range(gradient_steps):
                if step < learn_start or step < batch_size:
                    break

                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                batch_obs, actions, rewards, batch_next_obs, dones = map(lambda x: toTensor(x, self.device).float(),
                                                                         replay_buffer.sample(batch_size))

                if self.using_images:
                    batch_obs = self.encoder_net(channelFirst(batch_obs))
                    batch_next_obs = self.encoder_net(channelFirst(batch_next_obs))

                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)

                value_pred = self.value_net(batch_obs)
                q_value = self.q_value_net(batch_obs, actions)
                new_actions, log_pi, pre_tanh_value, mean_policy, logstd = self.policy.sampleAction(batch_obs)

                # Q-Value function loss
                target_value_pred = self.target_value_net(batch_next_obs)
                next_q_value = args.reward_scale * rewards + (1 - dones) * gamma * target_value_pred.detach()
                loss_q_value = 0.5 * q_value_criterion(q_value, next_q_value.detach())

                # Value Function loss
                q_value_new_actions = self.q_value_net(batch_obs, new_actions)
                next_value = q_value_new_actions - log_pi
                loss_value = 0.5 * value_criterion(value_pred, next_value.detach())

                # Policy Loss
                # why not log_pi.exp_() ?
                loss_policy = (log_pi.exp_() * (log_pi - q_value_new_actions + value_pred).detach()).mean()

                # pre_tanh_value
                loss_policy += w_reg * sum(map(l2Loss, [mean_policy, logstd]))

                optimizer.zero_grad()
                loss = loss_q_value + loss_value + loss_policy
                loss.backward()
                optimizer.step()

                # Update target value_pred network
                softUpdate(source=self.value_net, target=self.target_value_net, factor=soft_update_factor)

            if (step + 1) % print_freq == 0:
                print("{} steps - {:.2f} FPS".format(step, step / (time.time() - start_time)))


class PytorchPolicy(object):
    """
    The policy object for genetic algorithms, using Pytorch networks
    :param model: (Pytorch nn.Module) make sure there is no Sequential, as it breaks .shape function
    :param continuous_actions: (bool)
    :param srl_model: (bool) if using an srl model or not
    :param cuda: (bool)
    :param sampling: (bool) for sampling from the policy output, this makes the policy non-deterministic
    """

    def __init__(self, model, continuous_actions, device, srl_model=True, stochastic=True):
        super(PytorchPolicy, self).__init__()
        self.continuous_actions = continuous_actions
        self.model = model

    # used to prevent pickling of pytorch device object, as they cannot be pickled
    def __getstate__(self):
        d = self.__dict__.copy()
        d['model'] = d['model'].to(th.device('cpu'))
        if 'device' in d:
            d['device'] = 'cpu'
        return d

    # restore torch device from a pickle using the same config, if cuda is available
    def __setstate__(self, d):
        if 'device' in d:
            d['device'] = th.device("cuda" if th.cuda.is_available() and d['cuda'] else "cpu")
        d['model'] = d['model'].to(d['device'])
        self.__dict__.update(d)

    def sampleAction(self, obs):
        if self.continuous_actions:
            mean_policy, logstd = self.model(obs)
            # log_std_min=-20, log_std_max=2
            logstd = th.clamp(logstd, -20, 2)
            std = th.exp(logstd)
            distribution = Normal(mean_policy, std)
            pre_tanh_value = distribution.sample().detach()
            # Squash the value
            action = F.tanh(pre_tanh_value)
            # Correction to the log prob because of the squasing function
            epsilon = 1e-6
            log_pi = distribution.log_prob(pre_tanh_value) - th.log(1 - action ** 2 + epsilon)
            log_pi = log_pi.sum(-1, keepdim=True)
        else:
            mean_policy, logstd = self.model(obs)
            distribution = Categorical(logits=mean_policy)
            action = distribution.sample().detach()
            pre_tanh_value = action * 0.0
            logstd = logstd * 0.0
            log_pi = distribution.log_prob(action).unsqueeze(1)

        return action, log_pi, pre_tanh_value, mean_policy, logstd

    def getAction(self, obs):
        action, _, _, _, _ = self.sampleAction(obs)
        return detachToNumpy(action)
