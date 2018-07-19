import time
import pickle

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

from environments.utils import makeEnv
from rl_baselines.base_classes import BaseRLObject
from rl_baselines.utils import CustomVecNormalize, CustomDummyVecEnv, WrapFrameStack, \
    loadRunningAverage, MultiprocessSRLModel
from rl_baselines.models.sac_models import MLPPolicy, MLPQValueNetwork, MLPValueNetwork, NatureCNN
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
    :param device: (th.device)
    :return: (Tensor)
    """
    return th.from_numpy(arr).to(device)


def detachToNumpy(tensor):
    """
    Gets a pytorch tensor and returns a numpy array
    Detach creates a new Tensor,
    detached from the current graph whose node will never require gradient.
    :param tensor: (th.Tensor)
    :return: (numpy float)
    """
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


def channelFirst(tensor):
    """
    Permute the dimension to match pytorch convention
    for images (BCHW: Batch x Channel x Height x Width).
    :param tensor: (th.Tensor)
    :return: (th.Tensor)
    """
    return tensor.permute(0, 3, 1, 2)


class SACModel(BaseRLObject):
    """
    Class containing an implementation of soft actor critic
    Note: the policy with CNN on raw pixels is currenlty slow (5 FPS)
    Also, one difference with the paper is that the policy for continuous actions
    is a gaussian and not a mixture of gaussians.
    """

    def __init__(self):
        super(SACModel, self).__init__()
        self.device = None
        self.cuda = False
        self.policy_net, self.q_value_net, self.value_net, self.target_value_net = None, None, None, None
        self.deterministic = False  # Only available during testing
        self.continuous_actions = False
        self.encoder_net = None
        self.using_images = False
        # Min and max value for the std of the gaussian policy
        self.log_std_min = -20
        self.log_std_max = 2

    def save(self, save_path, _locals=None):
        assert self.policy_net is not None, "Error: must train or load model before use"
        with open(save_path, "wb") as f:
            pickle.dump(self.__getstate__(), f)
        # Move networks back to the right device
        self.__setstate__(self.__dict__)

    @classmethod
    def load(cls, load_path, args=None):
        with open(load_path, "rb") as f:
            class_dict = pickle.load(f)
        loaded_model = SACModel()
        loaded_model.__dict__ = class_dict
        return loaded_model

    @classmethod
    def makeEnv(cls, args, env_kwargs=None, load_path_normalise=None):
        # Even though SAC is single core only, we need to use the pipe system to work
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
        parser.add_argument('-lr', '--learning-rate', type=float, default=3e-4, help="Learning rate")
        parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
        parser.add_argument('--w-reg', type=float, default=1e-3, help="Weight for policy network regularization")
        parser.add_argument('--soft-update-factor', type=float, default=1e-2, help="Rate for updating target net weights")
        parser.add_argument('--print-freq', type=int, default=500, help="Print Frequency (every n steps)")
        parser.add_argument('--batch-size', type=int, default=128, help="Minibatch size for each gradient update")
        parser.add_argument('--gradient-steps', type=int, default=1, help="How many gradient update after each step")
        parser.add_argument('--reward-scale', type=float, default=1.0,
                            help="Scaling factor for raw reward. (entropy factor)")
        return parser

    def moveToDevice(self, device, d):
        """
        Move the different networks to a given device (cpu|cuda)
        :param device: (th.device)
        :param d: (dict) the class dictionnary
        """
        keys = ['value_net', 'target_value_net', 'q_value_net', 'policy_net']
        if self.using_images:
            keys += ['encoder_net']

        for key in keys:
            d[key] = d[key].to(device)

    # used to prevent pickling of pytorch device object, as they cannot be pickled
    def __getstate__(self):
        d = self.__dict__.copy()
        self.moveToDevice(th.device('cpu'), d)

        if 'device' in d:
            d['device'] = 'cpu'
        return d

    # restore torch device from a pickle using the same config, if cuda is available
    def __setstate__(self, d):
        if 'device' in d:
            d['device'] = th.device("cuda" if th.cuda.is_available() and d['cuda'] else "cpu")

        self.moveToDevice(d['device'], d)
        self.__dict__.update(d)

    def sampleAction(self, obs):
        """
        Sample action from Normal or Categorical distribution
        (continuous vs discrete actions) and return the log probability
        + policy parameters for regularization
        :param obs: (th.Tensor)
        :return: (tuple(th.Tensor))
        """
        if self.continuous_actions:
            mean_policy, log_std = self.policy_net(obs)
            # Clip the value of the standard deviation
            log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)
            std = th.exp(log_std)
            distribution = Normal(mean_policy, std)
            # Used only during testing
            if self.deterministic:
                pre_tanh_value = mean_policy
            else:
                pre_tanh_value = distribution.sample().detach()
            # Squash the value
            action = F.tanh(pre_tanh_value)
            # Correction to the log prob because of the squashing function
            epsilon = 1e-6
            log_pi = distribution.log_prob(pre_tanh_value) - th.log(1 - action ** 2 + epsilon)
            log_pi = log_pi.sum(-1, keepdim=True)
        else:
            mean_policy, log_std = self.policy_net(obs)
            # Here mean policy is the energy of each action
            distribution = Categorical(logits=mean_policy)
            if self.deterministic:
                action = th.argmax(F.softmax(mean_policy, dim=1), dim=1)
            else:
                action = distribution.sample().detach()
            # Only valid for continuous actions
            pre_tanh_value = action * 0.0
            log_std = log_std * 0.0
            log_pi = distribution.log_prob(action).unsqueeze(1)

        return action, log_pi, pre_tanh_value, mean_policy, log_std

    def encodeObservation(self, obs):
        """
        Convert observation to pytorch tensor
        and encode it (extract features) if needed using a CNN
        :param obs:(numpy array)
        :return: (th.Tensor)
        """
        obs = self.toFloatTensor(obs)
        if self.using_images:
            obs = self.encoder_net(channelFirst(obs))
        return obs

    def getActionProba(self, obs, dones=None):
        """
        Returns the action probability for the given observation
        :param obs: (numpy array)
        :param dones: ([bool])
        :return: (numpy float) the action probability
        """
        with th.no_grad():
            obs = self.encodeObservation(obs)
            mean_policy, _ = self.policy_net(obs)

            if self.continuous_actions:
                # In the case of continuous action
                # we return the mean of the gaussian policy
                # instead of probability
                action = mean_policy
            else:
                # In the case of discrete actions
                # mean_policy correspond to the energy|logits for each action
                # we need to apply a softmax in order to get a probability
                action = F.softmax(mean_policy, dim=-1)
        return detachToNumpy(action)

    def getAction(self, obs, dones=None):
        """
        From an observation returns the associated action
        :param obs: (numpy array)
        :param dones: ([bool])
        :return: (numpy float)
        """
        with th.no_grad():
            obs = self.encodeObservation(obs)
            action, _, _, _, _ = self.sampleAction(obs)

        return detachToNumpy(action)[0]

    def toFloatTensor(self, x):
        """
        Convert a numpy array to a torch float tensor
        :param x: (np.array)
        :return: (th.Tensor)
        """
        return toTensor(x, self.device).float()

    @classmethod
    def getOptParam(cls):
        return {
            "lr": (float, (1e-2, 1e-5)),
            "gamma": (float, (0, 1)),
            "w_reg": (float, (0, 1)),
            "soft_update_factor": (float, (0, 1)),
            "batch_size": (int, (32, 256)),
            "gradient_step": (int, (1, 10)),
            "reward_scale": (float, (0, 100))
        }

    def train(self, args, callback, env_kwargs=None, hyperparam=None):
        env = self.makeEnv(args, env_kwargs=env_kwargs)

        # set hyperparameters
        if hyperparam is not None:
            for name, val in hyperparam.items():
                args.__dict__["name"] = val

        self.cuda = th.cuda.is_available() and not args.no_cuda
        self.device = th.device("cuda" if self.cuda else "cpu")
        self.using_images = args.srl_model == "raw_pixels"
        self.continuous_actions = args.continuous_actions

        if args.continuous_actions:
            action_space = np.prod(env.action_space.shape)
        else:
            action_space = env.action_space.n

        if args.srl_model != "raw_pixels":
            input_dim = np.prod(env.observation_space.shape)
        else:
            n_channels = env.observation_space.shape[-1]
            # We use an additional CNN when using images
            # to extract features
            self.encoder_net = NatureCNN(n_channels).to(self.device)
            input_dim = 512  # output dim of the encoder net

        self.policy_net = MLPPolicy(input_dim, action_space).to(self.device)
        self.q_value_net = MLPQValueNetwork(input_dim, action_space, args.continuous_actions).to(self.device)
        self.value_net = MLPValueNetwork(input_dim).to(self.device)
        self.target_value_net = MLPValueNetwork(input_dim).to(self.device)

        # Make sure target net has the same weights as value_net
        hardUpdate(source=self.value_net, target=self.target_value_net)

        value_criterion = nn.MSELoss()
        q_value_criterion = nn.MSELoss()

        replay_buffer = ReplayBuffer(args.buffer_size)

        policy_optimizer = th.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate)
        value_optimizer = th.optim.Adam(self.value_net.parameters(), lr=args.learning_rate)
        q_optimizer = th.optim.Adam(self.q_value_net.parameters(), lr=args.learning_rate)

        obs = env.reset()
        start_time = time.time()

        for step in range(args.num_timesteps):
            action = self.getAction(obs[None])
            new_obs, reward, done, info = env.step(action)

            # Fill the replay buffer
            replay_buffer.add(obs, action, reward, new_obs, float(done))
            obs = new_obs

            # Callback for plotting and saving best model
            if callback is not None:
                callback(locals(), globals())

            if done:
                obs = env.reset()

            # Update the different networks
            for _ in range(args.gradient_steps):
                # Check that there is enough data in the buffer replay
                if step < args.batch_size:
                    break

                # Sample a minibatch from the replay buffer
                batch_obs, actions, rewards, batch_next_obs, dones = map(lambda x: self.toFloatTensor(x),
                                                                         replay_buffer.sample(args.batch_size))

                if self.using_images:
                    # Extract features from the images
                    batch_obs = self.encoder_net(channelFirst(batch_obs))
                    batch_next_obs = self.encoder_net(channelFirst(batch_next_obs))

                rewards = rewards.unsqueeze(1)
                dones = dones.unsqueeze(1)

                value_pred = self.value_net(batch_obs)
                q_value = self.q_value_net(batch_obs, actions)
                # Sample actions and retrieve log proba
                # pre_tanh_value, mean_policy and log_std are only used for regularization
                new_actions, log_pi, pre_tanh_value, mean_policy, log_std = self.sampleAction(batch_obs)

                # Q-Value function loss
                target_value_pred = self.target_value_net(batch_next_obs)
                # TD error with reward scaling
                next_q_value = args.reward_scale * rewards + (1 - dones) * args.gamma * target_value_pred.detach()
                loss_q_value = 0.5 * q_value_criterion(q_value, next_q_value.detach())

                # Value Function loss
                q_value_new_actions = self.q_value_net(batch_obs, new_actions)
                next_value = q_value_new_actions - log_pi
                loss_value = 0.5 * value_criterion(value_pred, next_value.detach())

                # Policy Loss
                # why not log_pi.exp_() ?
                loss_policy = (log_pi * (log_pi - q_value_new_actions + value_pred).detach()).mean()
                # Regularization
                if self.continuous_actions:
                    loss_policy += args.w_reg * sum(map(l2Loss, [mean_policy, log_std]))

                q_optimizer.zero_grad()
                # Retain graph if we are using a CNN for extracting features
                loss_q_value.backward(retain_graph=self.using_images)
                q_optimizer.step()

                value_optimizer.zero_grad()
                loss_value.backward(retain_graph=self.using_images)
                value_optimizer.step()

                policy_optimizer.zero_grad()
                loss_policy.backward()
                policy_optimizer.step()

                # Softly update target value_pred network
                softUpdate(source=self.value_net, target=self.target_value_net, factor=args.soft_update_factor)

            if (step + 1) % args.print_freq == 0:
                print("{} steps - {:.2f} FPS".format(step, step / (time.time() - start_time)))
