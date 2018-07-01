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
from srl_zoo.utils import printYellow


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
    with th.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.copy_(
                source_param * factor + target_param * (1.0 - factor)
            )


def hardUpdate(*, source, target):
    with th.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.copy_(source_param)


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
        parser.add_argument('--cuda', action='store_true', default=False,
                            help='use gpu for the neural network')
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

        if args.continuous_actions:
            action_space = np.prod(env.action_space.shape)
        else:
            action_space = env.action_space.n

        if args.srl_model != "raw_pixels":
            printYellow("Using MLP policy because working on state representation")
            args.policy = "mlp"
            input_dim = np.prod(env.observation_space.shape)
            self.policy_net = MLPPolicy(input_dim, [100], action_space)
            self.q_value_net = MLPQValueNetwork(input_dim, action_space)
            self.value_net = MLPValueNetwork(input_dim)
            self.target_value_net = MLPValueNetwork(input_dim)
        else:
            raise ValueError()
            # net = CNNPolicyPytorch(env.observation_space.shape[-1], action_space)

        self.device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

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
        batch_size = 64
        replay_buffer = ReplayBuffer(args.buffer_size)

        policy_optimizer = th.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        value_optimizer = th.optim.Adam(self.value_net.parameters(), lr=learning_rate)
        q_optimizer = th.optim.Adam(self.q_value_net.parameters(), lr=learning_rate)

        obs = env.reset()
        start_time = time.time()

        for step in range(args.num_timesteps):
            action = self.policy.getAction(toTensor(obs[None], self.device))[0]
            new_obs, reward, done, info = env.step(action)

            replay_buffer.add(obs, action, reward, new_obs, float(done))
            obs = new_obs

            if callback is not None:
                callback(locals(), globals())

            if done:
                obs = env.reset()

            for _ in range(gradient_steps):
                if step < batch_size:
                    break
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                batch_obs, actions, rewards, batch_next_obs, dones = map(lambda x: toTensor(x, self.device),
                                                                         replay_buffer.sample(batch_size))

                value = self.value_net(batch_obs)
                q_value = self.q_value_net(batch_obs, actions)
                new_actions, log_prob, z, mu, logvar = self.policy.sampleAction(batch_obs)

                target_value = self.target_value_net(batch_next_obs)
                next_q_value = args.reward_scale * reward + (1 - done) * gamma * target_value.detach()
                loss_q_value = 0.5 * q_value_criterion(q_value, next_q_value.detach())

                new_q_value = self.q_value_net(batch_obs, new_actions)
                next_value = new_q_value - log_prob
                # loss_v = 0.5 * (vf_t - (log_target - log_pi + policy_prior_log_probs)) **  2
                loss_value = 0.5 * value_criterion(value, next_value.detach())

                # grad_pi = grad_log_prob(action) * (log_prob(action) + 1 - q_value(obs, action) + value(obs))
                # policy_kl_loss = tf.reduce_mean(log_pi * tf.stop_gradient(log_pi - log_target + self._vf_t - policy_prior_log_probs))
                log_prob_target = new_q_value - value
                loss_policy = (log_prob * (log_prob - log_prob_target).detach()).mean()

                w_reg = 1e-3
                mean_loss = w_reg * mu.pow(2).mean()
                std_loss = w_reg * logvar.pow(2).mean()
                z_loss = w_reg * z.pow(2).sum(1).mean()

                loss_policy += mean_loss + std_loss + z_loss

                q_optimizer.zero_grad()
                loss_q_value.backward()
                q_optimizer.step()

                value_optimizer.zero_grad()
                loss_value.backward()
                value_optimizer.step()

                policy_optimizer.zero_grad()
                loss_policy.backward()
                policy_optimizer.step()

                # Update target value network
                softUpdate(source=self.value_net, target=self.target_value_net, factor=0.01)

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
        self.srl_model = srl_model
        self.stochastic = stochastic
        self.device = device

        self.model = self.model.to(self.device)

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
        mu, logvar = self.model(obs)
        # logvar = \log(\sigma^2) = 2 * \log(\sigma)
        # \sigma = \exp(0.5 * logvar)
        std = logvar.mul(0.5).exp_()
        distribution = Normal(mu, std)
        z = distribution.sample()
        # Squash the value
        action = F.tanh(z)
        # Correction to the log prob
        log_prob = distribution.log_prob(z) - th.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        # action = z.detach()
        # log_prob = distribution.log_prob(action)

        return action, log_prob, z, mu, logvar

    def getAction(self, obs):
        action, _, _, _, _ = self.sampleAction(obs)
        return detachToNumpy(action)

    # def selectAction(self, obs):
    #     logits = self.model(toTensor(obs, self.device))
    #     distribution = Categorical(logits=logits)
    #     action = distribution.sample()
    #     return action, -distribution.log_prob(action)


class MLPPolicy(nn.Module):
    """
    :param input_dim: (int)
    :param hidden_dims: ([int])
    :param out_dim: (int)
    """

    def __init__(self, input_dim, hidden_dims, out_dim):
        super(MLPPolicy, self).__init__()

        self.policy_net = nn.Sequential(
            nn.Linear(int(input_dim), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.mean_head = nn.Linear(128, int(out_dim))
        self.logvar_head = nn.Linear(128, int(out_dim))

    def forward(self, x):
        x = self.policy_net(x)
        # log_var = th.clamp(log_var, self.log_var_min, self.log_var_max)
        return self.mean_head(x), self.logvar_head(x)


class MLPValueNetwork(nn.Module):
    """
    :param input_dim: (int)
    """

    def __init__(self, input_dim):
        super(MLPValueNetwork, self).__init__()

        self.value_net = nn.Sequential(
            nn.Linear(int(input_dim), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.value_net(x)


class MLPQValueNetwork(nn.Module):
    """
    :param input_dim: (int)
    """

    def __init__(self, input_dim, n_actions):
        super(MLPQValueNetwork, self).__init__()

        self.q_value_net = nn.Sequential(
            nn.Linear(int(input_dim) + int(n_actions), 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, obs, action):
        return self.q_value_net(th.cat([obs, action], dim=1))
